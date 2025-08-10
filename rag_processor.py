import os
import tempfile
import requests
import gc
import re
from pypdf import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class RAGProcessor:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,  # Lower temperature for more consistent answers
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

    def load_pdf_from_url(self, url):
        """Download a PDF from remote URL and return text."""
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type:
            raise ValueError(f"URL did not return a PDF. Got: {content_type}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        text = ""
        reader = PdfReader(tmp_path)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            # Basic text cleaning
            page_text = re.sub(r'\s+', ' ', page_text)  # Normalize whitespace
            text += page_text + "\n"
        
        # Clean up temporary file
        os.unlink(tmp_path)
        return text.strip()

    def split_text(self, text):
        # Smart chunking with overlap
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  # Slightly larger for better context
            chunk_overlap=150,  # Optimized overlap
            separators=["\n\n", "\n", ". ", ".", " ", ""]  # Better splitting
        )
        docs = splitter.split_text(text)
        return [Document(page_content=chunk.strip()) for chunk in docs if len(chunk.strip()) > 50]

    def create_vectorstore(self, documents):
        # Batch embedding creation for efficiency
        doc_texts = [doc.page_content for doc in documents]
        
        # Create embeddings in smaller batches to save memory
        batch_size = 10
        doc_embeddings = []
        
        for i in range(0, len(doc_texts), batch_size):
            batch = doc_texts[i:i+batch_size]
            batch_embeddings = []
            
            for text in batch:
                embedding = self.embeddings.embed_query(text)
                batch_embeddings.append(embedding)
            
            doc_embeddings.extend(batch_embeddings)
            
            # Memory cleanup after each batch
            gc.collect()
        
        return {
            'documents': documents,
            'embeddings': np.array(doc_embeddings, dtype=np.float32),  # Use float32 to save memory
            'texts': doc_texts
        }

    def retrieve_documents(self, vectorstore, query, k=3):  # Increased to 3 for better accuracy
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, vectorstore['embeddings'])[0]
        
        # Get top k most similar documents with threshold
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Filter out low similarity documents (below 0.3 threshold)
        relevant_docs = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Only include relevant docs
                relevant_docs.append(vectorstore['documents'][idx])
        
        return relevant_docs if relevant_docs else [vectorstore['documents'][top_indices[0]]]

    def run_rag(self, doc_url, questions):
        # Step 1: Download and load document
        text = self.load_pdf_from_url(doc_url)

        # Step 2: Split into chunks
        docs = self.split_text(text)

        # Step 3: Embed + store (only once for all questions)
        vectorstore = self.create_vectorstore(docs)

        # Step 4: Retrieve and answer
        answers = []
        for q in questions:
            rel_docs = self.retrieve_documents(vectorstore, q, k=3)
            context = "\n\n".join([d.page_content for d in rel_docs])
            
            # Enhanced prompt for better accuracy
            prompt = f"""You are a precise document analyst. Based ONLY on the provided context, answer the question with specific details from the document. Be direct and factual.

If the information is not in the context, respond with "Information not available in the provided document."

Context:
{context}

Question: {q}

Provide a clear, specific answer:"""
            
            res = self.llm.predict(prompt)
            
            # Enhanced text cleaning
            clean_answer = res.strip()
            clean_answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_answer)  # Remove bold formatting
            clean_answer = re.sub(r'#{1,6}\s*', '', clean_answer)  # Remove headers
            clean_answer = re.sub(r'\n{3,}', '. ', clean_answer)  # Multiple newlines to period
            clean_answer = re.sub(r'\n{2}', '. ', clean_answer)   # Double newlines to period
            clean_answer = re.sub(r'\n', ' ', clean_answer)       # Single newlines to space
            clean_answer = re.sub(r'\s{2,}', ' ', clean_answer)   # Multiple spaces to single
            clean_answer = clean_answer.strip()
            
            answers.append(clean_answer)
            
            # Memory cleanup after each question
            del res, clean_answer
            gc.collect()

        # Final cleanup
        del vectorstore
        gc.collect()
        
        return answers