import os
import tempfile
import requests
import gc
from typing import List
import mimetypes

# Document loaders
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredEmailLoader

# Lightweight embeddings and text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# Simple FAISS for vector storage (more memory efficient)
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    FAISS_AVAILABLE = True
except ImportError:
    from langchain_core.vectorstores import InMemoryVectorStore
    FAISS_AVAILABLE = False


class RAGProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = None  # Lazy load
        self.embeddings = None  # Lazy load
        
        # Simple RAG prompt template
        self.prompt_template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise and accurate.

Context:
{context}

Question: {question}

Answer:"""

    def _get_llm(self):
        """Lazy load LLM to save memory"""
        if self.llm is None:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",  # Faster and cheaper than gemini-pro
                google_api_key=self.api_key,
                temperature=0.1
            )
        return self.llm

    def _get_embeddings(self):
        """Lazy load embeddings with smallest model"""
        if self.embeddings is None:
            # Use smallest possible embedding model
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},  # Force CPU to save GPU memory
                encode_kwargs={'batch_size': 8}   # Small batch size
            )
        return self.embeddings

    def _detect_file_type(self, url: str) -> str:
        """Detect file type from URL or download a small chunk"""
        # Try to guess from URL first
        mime_type, _ = mimetypes.guess_type(url)
        if mime_type:
            if 'pdf' in mime_type:
                return 'pdf'
            elif 'word' in mime_type or 'docx' in mime_type:
                return 'docx'
            elif 'email' in mime_type or url.endswith('.eml'):
                return 'eml'
        
        # If unclear, download first few bytes to check
        try:
            response = requests.get(url, headers={'Range': 'bytes=0-1023'}, timeout=10)
            content = response.content
            if content.startswith(b'%PDF'):
                return 'pdf'
            elif b'PK' in content[:4]:  # ZIP-based formats like DOCX
                return 'docx'
            elif b'Return-Path:' in content or b'Message-ID:' in content:
                return 'eml'
        except:
            pass
        
        return 'pdf'  # Default assumption

    def _download_file(self, url: str) -> str:
        """Download file to temporary location"""
        file_type = self._detect_file_type(url)
        suffix_map = {'pdf': '.pdf', 'docx': '.docx', 'eml': '.eml'}
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix_map.get(file_type, '.pdf')) as tmp_file:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                raise Exception(f"Failed to download file: {response.status_code}")
            tmp_file.write(response.content)
            return tmp_file.name, file_type

    def _load_document(self, file_path: str, file_type: str):
        """Load document based on file type"""
        try:
            if file_type == 'pdf':
                loader = PyPDFLoader(file_path)
            elif file_type == 'docx':
                loader = Docx2txtLoader(file_path)
            elif file_type == 'eml':
                loader = UnstructuredEmailLoader(file_path)
            else:
                # Fallback to PDF
                loader = PyPDFLoader(file_path)
            
            return loader.load()
        except Exception as e:
            # If specific loader fails, try text extraction
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                from langchain_core.documents import Document
                return [Document(page_content=content)]
            except:
                raise Exception(f"Could not load document: {str(e)}")

    def process_document_from_url(self, doc_url: str, questions: List[str]) -> List[str]:
        temp_file = None
        try:
            # === Step 1: Download file ===
            temp_file, file_type = self._download_file(doc_url)

            # === Step 2: Load document ===
            docs = self._load_document(temp_file, file_type)

            # === Step 3: Split into smaller chunks to save memory ===
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,      # Smaller chunks
                chunk_overlap=50,    # Smaller overlap
                length_function=len
            )
            pages = splitter.split_documents(docs)
            
            # Clear original docs to free memory
            del docs
            gc.collect()

            # === Step 4: Create vector store (memory efficient) ===
            embeddings = self._get_embeddings()
            
            if FAISS_AVAILABLE:
                # FAISS is more memory efficient for larger datasets
                vector_store = FAISS.from_documents(pages, embeddings)
                retriever = vector_store.as_retriever(
                    search_kwargs={"k": 3}  # Retrieve fewer documents
                )
            else:
                from langchain_core.vectorstores import InMemoryVectorStore
                vector_store = InMemoryVectorStore.from_documents(pages, embeddings)
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})

            # Clear pages to free memory
            del pages
            gc.collect()

            # === Step 5: Process questions ===
            llm = self._get_llm()
            parser = StrOutputParser()
            
            answers = []
            for question in questions:
                try:
                    # Get relevant documents
                    relevant_docs = retriever.invoke(question)
                    
                    # Format context
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    
                    # Create prompt
                    prompt = self.prompt_template.format(context=context, question=question)
                    
                    # Get answer
                    response = llm.invoke(prompt)
                    answer = parser.invoke(response)
                    answers.append(answer.strip())
                    
                    # Force garbage collection between questions
                    gc.collect()
                    
                except Exception as e:
                    answers.append(f"Error processing question: {str(e)}")

            return answers

        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
            
            # Force final garbage collection
            gc.collect()