import os
import tempfile
import requests
import gc
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
            temperature=0.3,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

    def load_pdf_from_url(self, url):
        """Download a PDF from remote URL and return text."""
        resp = requests.get(url, stream=True)
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
            text += page.extract_text() or ""
        
        # Clean up temporary file
        os.unlink(tmp_path)
        return text

    def split_text(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = splitter.split_text(text)
        return [Document(page_content=chunk) for chunk in docs]

    def create_vectorstore(self, documents):
        # Create embeddings for all documents
        doc_texts = [doc.page_content for doc in documents]
        doc_embeddings = []
        for text in doc_texts:
            embedding = self.embeddings.embed_query(text)
            doc_embeddings.append(embedding)
        
        return {
            'documents': documents,
            'embeddings': np.array(doc_embeddings),
            'texts': doc_texts
        }

    def retrieve_documents(self, vectorstore, query, k=2):
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, vectorstore['embeddings'])[0]
        
        # Get top k most similar documents
        top_indices = np.argsort(similarities)[::-1][:k]
        
        return [vectorstore['documents'][i] for i in top_indices]

    def run_rag(self, doc_url, questions):
        # Step 1: Download and load document
        text = self.load_pdf_from_url(doc_url)

        # Step 2: Split into chunks
        docs = self.split_text(text)

        # Step 3: Embed + store
        vectorstore = self.create_vectorstore(docs)

        # Step 4: Retrieve and answer
        answers = []
        for q in questions:
            rel_docs = self.retrieve_documents(vectorstore, q, k=2)
            context = "\n\n".join([d.page_content for d in rel_docs])
            
            prompt = f"""Based on the provided document context, answer the question accurately and concisely in plain text. Do not use markdown formatting, bullet points, or special characters like ** or ##. Provide a direct, clean answer.

Context:
{context}

Question: {q}

Answer:"""
            
            res = self.llm.predict(prompt)
            # Clean up markdown formatting but keep readability
            clean_answer = res.strip()
            clean_answer = clean_answer.replace('**', '')  # Remove bold
            clean_answer = clean_answer.replace('##', '')  # Remove headers
            clean_answer = clean_answer.replace('\n\n\n', '. ')  # Triple newlines to period+space
            clean_answer = clean_answer.replace('\n\n', '. ')   # Double newlines to period+space
            clean_answer = clean_answer.replace('\n', ' ')      # Single newlines to space
            answers.append(clean_answer)
            
            # Force garbage collection to free memory
            gc.collect()

        return answers