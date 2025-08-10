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
            temperature=0.0,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

    def clean_text(self, text):
        """Enhanced text cleaning with better structure preservation"""
        # Fix unicode issues
        text = unicodedata.normalize('NFKD', text)
        
        # Replace common unicode characters
        replacements = {
            '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"',
            '\u2013': '-', '\u2014': '-', '\u00a0': ' ', '\u2022': '*',
            '\u2026': '...', '\u00b7': '*', '\u25cf': '*'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove unwanted phrases
        unwanted_phrases = [
            '"Answer is not included"', "'Answer is not included'",
            "Answer is not included", "\\\"Answer is not included\\\"",
            '\\"Answer is not included\\"'
        ]
        
        for phrase in unwanted_phrases:
            text = text.replace(phrase, "")
        
        # Preserve important structure while removing markdown
        text = re.sub(r'#{1,6}\s*([^\n]+)', r'\1:', text)  # Headers to colons
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)     # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)         # Italic
        text = re.sub(r'```[^`]*```', '', text)            # Code blocks
        text = re.sub(r'`([^`]+)`', r'\1', text)           # Inline code
        
        # Better paragraph handling - preserve important breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)     # Multiple newlines to double
        text = re.sub(r'(?<=[.!?])\s*\n(?=[A-Z])', ' ', text)  # Join sentences split across lines
        text = re.sub(r'(?<=[a-z,])\s*\n(?=[a-z])', ' ', text)  # Join mid-sentence breaks
        
        # Clean up formatting
        text = text.replace('\\', '')
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\'\"\(\)\[\]]', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        
        return text.strip()

    def load_pdf_from_url(self, url):
        """Download PDF with better error handling"""
        try:
            resp = requests.get(url, stream=True, timeout=20, 
                              headers={'User-Agent': 'RAG-Processor/1.0'})
            resp.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                for chunk in resp.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                tmp_path = tmp.name

            text = ""
            reader = PdfReader(tmp_path)
            
            # Extract with better page handling
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text += f"\n--- Page {i+1} ---\n{page_text}"
                except Exception as e:
                    print(f"Error extracting page {i+1}: {e}")
                    continue
            
            os.unlink(tmp_path)
            return self.clean_text(text)
            
        except Exception as e:
            raise Exception(f"Failed to load PDF: {str(e)}")

    def split_text(self, text):
        """Enhanced splitting with semantic awareness"""
        # Use multiple splitters for better chunking
        splitters = [
            RecursiveCharacterTextSplitter(
                chunk_size=2000,  # Larger chunks for more context
                chunk_overlap=300,
                separators=["\n--- Page", "\n\n", "\n", ". ", ".", " "]
            ),
            # Backup splitter for very long sections
            RecursiveCharacterTextSplitter(
                chunk_size=1800,
                chunk_overlap=250,
                separators=[". ", ".", " "]
            )
        ]
        
        docs = []
        for splitter in splitters:
            try:
                chunks = splitter.split_text(text)
                docs = [Document(page_content=chunk.strip(), 
                               metadata={'chunk_id': i}) 
                       for i, chunk in enumerate(chunks) 
                       if len(chunk.strip()) > 100]
                if docs:  # If we got good chunks, use them
                    break
            except:
                continue
        
        return docs

    def create_vectorstore_parallel(self, documents):
        """Memory-efficient embedding creation with deduplication"""
        # Deduplicate similar chunks to save memory
        unique_docs = self._deduplicate_chunks(documents)
        doc_texts = [doc.page_content for doc in unique_docs]
        
        # Process in tiny batches for memory safety
        batch_size = 3
        doc_embeddings = []
        
        print(f"Creating embeddings for {len(doc_texts)} chunks...")
        
        for i in range(0, len(doc_texts), batch_size):
            batch = doc_texts[i:i+batch_size]
            batch_embeddings = []
            
            for text in batch:
                try:
                    embedding = self.embeddings.embed_query(text)
                    batch_embeddings.append(np.array(embedding, dtype=np.float32))
                except Exception as e:
                    print(f"Embedding error: {e}")
                    # Use zero vector as fallback
                    batch_embeddings.append(np.zeros(768, dtype=np.float32))
            
            doc_embeddings.extend(batch_embeddings)
            
            # Aggressive garbage collection
            del batch_embeddings
            gc.collect()
            
            if i % 15 == 0:  # Progress indicator
                print(f"Progress: {i+batch_size}/{len(doc_texts)}")
        
        embeddings_array = np.vstack(doc_embeddings)
        del doc_embeddings
        gc.collect()
        
        return {
            'documents': unique_docs,
            'embeddings': embeddings_array,
            'texts': doc_texts
        }

    def _deduplicate_chunks(self, documents, similarity_threshold=0.85):
        """Remove highly similar chunks to save memory and improve quality"""
        if len(documents) <= 10:
            return documents
        
        unique_docs = [documents[0]]
        
        for doc in documents[1:]:
            is_similar = False
            doc_text = doc.page_content.lower()
            
            # Quick text-based similarity check
            for unique_doc in unique_docs:
                unique_text = unique_doc.page_content.lower()
                
                # Simple overlap ratio
                words1 = set(doc_text.split())
                words2 = set(unique_text.split())
                
                if len(words1) > 0 and len(words2) > 0:
                    overlap = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    
                    if overlap / union > similarity_threshold:
                        is_similar = True
                        break
            
            if not is_similar:
                unique_docs.append(doc)
        
        print(f"Deduplicated: {len(documents)} -> {len(unique_docs)} chunks")
        return unique_docs

    def retrieve_documents(self, vectorstore, query, k=6):
        """Enhanced retrieval with multiple strategies"""
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # Compute similarities
        similarities = cosine_similarity(query_embedding, vectorstore['embeddings'])[0]
        
        # Multi-stage retrieval
        top_indices = np.argsort(similarities)[::-1]
        
        # Strategy 1: High similarity docs
        high_sim_docs = []
        for idx in top_indices:
            if similarities[idx] > 0.35 and len(high_sim_docs) < k//2:
                high_sim_docs.append((vectorstore['documents'][idx], similarities[idx]))
        
        # Strategy 2: Diverse retrieval - avoid too similar docs
        diverse_docs = []
        used_indices = set([idx for idx, _ in [(i, 0) for i, (doc, sim) in enumerate(high_sim_docs)]])
        
        for idx in top_indices:
            if len(diverse_docs) >= k - len(high_sim_docs):
                break
                
            if idx in used_indices:
                continue
                
            # Check if this doc is too similar to already selected ones
            doc_text = vectorstore['documents'][idx].page_content.lower()
            is_too_similar = False
            
            for selected_doc, _ in high_sim_docs + diverse_docs:
                selected_text = selected_doc.page_content.lower()
                words1 = set(doc_text.split()[:50])  # First 50 words
                words2 = set(selected_text.split()[:50])
                
                if len(words1.intersection(words2)) / max(len(words1), len(words2), 1) > 0.6:
                    is_too_similar = True
                    break
            
            if not is_too_similar and similarities[idx] > 0.15:
                diverse_docs.append((vectorstore['documents'][idx], similarities[idx]))
                used_indices.add(idx)
        
        # Combine and sort by similarity
        all_docs = high_sim_docs + diverse_docs
        all_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top documents
        final_docs = [doc for doc, sim in all_docs[:k]]
        
        # Ensure we have at least 2 docs
        if len(final_docs) < 2:
            final_docs = [vectorstore['documents'][i] for i in top_indices[:2]]
        
        return final_docs

    def run_rag(self, doc_url, questions):
        """Enhanced RAG pipeline with better accuracy"""
        print(f"Processing document: {doc_url}")
        
        # Step 1: Download and load document
        text = self.load_pdf_from_url(doc_url)
        print(f"Extracted text length: {len(text)} characters")

        # Step 2: Split into chunks
        docs = self.split_text(text)
        print(f"Created {len(docs)} chunks")

        # Step 3: Create embeddings
        vectorstore = self.create_vectorstore_parallel(docs)

        # Step 4: Process questions with enhanced retrieval
        answers = []
        total_questions = len(questions)
        
        for i, q in enumerate(questions):
            print(f"Processing question {i+1}/{total_questions}")
            
            # Retrieve relevant documents
            rel_docs = self.retrieve_documents(vectorstore, q, k=5)
            
            # Create comprehensive context
            context_parts = []
            for j, doc in enumerate(rel_docs):
                context_parts.append(f"Passage {j+1}:\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)
            
            # Enhanced prompt for better accuracy
            prompt = f"""You are analyzing a document to answer questions accurately. Use the provided context passages to give precise, factual answers.

Context:
{context}

Question: {q}

Instructions:
- Answer directly based on the context provided
- If the answer spans multiple passages, synthesize the information
- Be concise but complete (2-3 sentences max)
- If the context doesn't contain enough information, state what you can determine and note any limitations

Answer:"""
            
            try:
                response = self.llm.predict(prompt)
                cleaned_answer = self.clean_text(response)
                answers.append(cleaned_answer)
            except Exception as e:
                print(f"Error processing question {i+1}: {e}")
                answers.append("Error processing this question.")
            
            # Memory cleanup
            if i % 5 == 0:
                gc.collect()
        
        # Final cleanup
        del vectorstore, docs, text
        gc.collect()
        
        print("RAG processing completed")
        return answers