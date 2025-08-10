import os
import tempfile
import requests
import gc
import re
import unicodedata
import time
from concurrent.futures import ThreadPoolExecutor
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
            model="gemini-2.5-flash",
            temperature=0.0,  # Zero temperature for maximum consistency
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

    def clean_text(self, text):
        """Comprehensive text cleaning"""
        # Fix unicode issues
        text = unicodedata.normalize('NFKD', text)
        
        # Replace common unicode characters
        replacements = {
            '\u2019': "'",  # Right single quotation mark
            '\u2018': "'",  # Left single quotation mark  
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u2013': '-',  # En dash
            '\u2014': '-',  # Em dash
            '\u00a0': ' ',  # Non-breaking space
            '\u2022': '*',  # Bullet point
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove unwanted phrases
        unwanted_phrases = [
            '"Answer is not included"',
            "'Answer is not included'",
            "Answer is not included",
            "\\\"Answer is not included\\\"",
            '\\"Answer is not included\\"'
        ]
        
        for phrase in unwanted_phrases:
            text = text.replace(phrase, "")
        
        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic
        text = re.sub(r'#{1,6}\s*', '', text)           # Headers
        text = re.sub(r'```[^`]*```', '', text)         # Code blocks
        text = re.sub(r'`([^`]+)`', r'\1', text)        # Inline code
        
        # Fix spacing and newlines
        text = re.sub(r'\n{3,}', '. ', text)           # Multiple newlines
        text = re.sub(r'\n{2}', '. ', text)            # Double newlines
        text = re.sub(r'\n', ' ', text)                # Single newlines
        text = re.sub(r'\s{2,}', ' ', text)            # Multiple spaces
        
        # Remove backslashes and escape sequences
        text = text.replace('\\', '')
        text = text.replace('\\"', '"')
        text = text.replace("\\'", "'")
        
        # Clean up extra punctuation and weird characters
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\'\"]', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)  # Final space cleanup
        
        return text.strip()

    def load_pdf_from_url(self, url):
        """Download a PDF from remote URL and return text."""
        resp = requests.get(url, stream=True, timeout=15)  # Reduced timeout
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        text = ""
        reader = PdfReader(tmp_path)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        
        os.unlink(tmp_path)
        return self.clean_text(text)

    def split_text(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Larger chunks for better context
            chunk_overlap=200,  # More overlap for continuity
            separators=["\n\n", "\n", ". ", ".", " "]
        )
        docs = splitter.split_text(text)
        return [Document(page_content=chunk.strip()) for chunk in docs if len(chunk.strip()) > 50]

    def create_vectorstore_parallel(self, documents):
        """Create embeddings with controlled memory usage"""
        doc_texts = [doc.page_content for doc in documents]
        
        # Process in smaller batches to control memory
        batch_size = 6  # Smaller batches
        doc_embeddings = []
        
        for i in range(0, len(doc_texts), batch_size):
            batch = doc_texts[i:i+batch_size]
            
            # Single-threaded within batch for memory safety
            batch_embeddings = []
            for text in batch:
                embedding = self.embeddings.embed_query(text)
                batch_embeddings.append(embedding)
            
            doc_embeddings.extend(batch_embeddings)
            gc.collect()  # Clean up after each batch
        
        return {
            'documents': documents,
            'embeddings': np.array(doc_embeddings, dtype=np.float32),
            'texts': doc_texts
        }

    def retrieve_documents(self, vectorstore, query, k=4):  # More docs for better accuracy
        query_embedding = self.embeddings.embed_query(query)
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        similarities = cosine_similarity(query_embedding, vectorstore['embeddings'])[0]
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Filter for relevance
        relevant_docs = []
        for idx in top_indices:
            if similarities[idx] > 0.25:  # Lower threshold for more docs
                relevant_docs.append(vectorstore['documents'][idx])
        
        # Return at least 2 docs even if below threshold
        if len(relevant_docs) < 2:
            return [vectorstore['documents'][i] for i in top_indices[:2]]
        
        return relevant_docs

    def run_rag(self, doc_url, questions):
        # Step 1: Download and load document
        text = self.load_pdf_from_url(doc_url)

        # Step 2: Split into chunks
        docs = self.split_text(text)

        # Step 3: Create embeddings in parallel
        vectorstore = self.create_vectorstore_parallel(docs)

        # Step 4: Process questions sequentially for memory safety
        answers = []
        for q in questions:
            rel_docs = self.retrieve_documents(vectorstore, q, k=2)
            context = "\n\n".join([d.page_content for d in rel_docs])
            
            # Concise prompt for shorter answers
            prompt = f"""Based on the document context, provide a direct, concise answer. Use only 1-2 sentences maximum.

Context: {context}

Question: {q}

Concise answer:"""
            
            res = self.llm.predict(prompt)
            answers.append(self.clean_text(res))
            
            # Memory cleanup after each question
            gc.collect()
        
        # Cleanup
        del vectorstore
        gc.collect()
        
        return answers