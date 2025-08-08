import os
import requests
import gc
from typing import List
import email
from email.parser import BytesParser
from email.policy import default
import io
import numpy as np

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class SimpleEmbeddings:
    def __init__(self):
        if FAISS_AVAILABLE:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.model = None
    
    def embed_documents(self, texts):
        if self.model:
            return self.model.encode(texts, convert_to_tensor=False).tolist()
        return [[0.0] * 384 for _ in texts]
    
    def embed_query(self, text):
        if self.model:
            return self.model.encode([text], convert_to_tensor=False)[0].tolist()
        return [0.0] * 384


class RAGProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = None
        self.embeddings = None
        self.prompt_template = """You are an assistant for question-answering tasks...
Context:
{context}

Question: {question}

Answer:"""

    def _get_llm(self):
        if self.llm is None:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=self.api_key,
                temperature=0.1
            )
        return self.llm

    def _get_embeddings(self):
        if self.embeddings is None:
            self.embeddings = SimpleEmbeddings()
        return self.embeddings

    def _detect_file_type(self, url: str) -> str:
        url_lower = url.lower()
        if '.pdf' in url_lower:
            return 'pdf'
        elif '.docx' in url_lower:
            return 'docx'
        elif '.eml' in url_lower:
            return 'eml'
        try:
            response = requests.head(url, timeout=5)
            ctype = response.headers.get('content-type', '').lower()
            if 'pdf' in ctype:
                return 'pdf'
            elif 'word' in ctype or 'docx' in ctype:
                return 'docx'
        except:
            pass
        return 'pdf'

    def _extract_text_from_pdf(self, file_content: bytes) -> str:
        if not PDF_AVAILABLE:
            raise Exception("PyPDF2 not available")
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            return "\n".join(page.extract_text() for page in pdf_reader.pages)
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {e}")

    def _extract_text_from_docx(self, file_content: bytes) -> str:
        if not DOCX_AVAILABLE:
            raise Exception("python-docx not available")
        try:
            docx_file = io.BytesIO(file_content)
            doc = DocxDocument(docx_file)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            raise Exception(f"Error extracting DOCX text: {e}")

    def _extract_text_from_eml(self, file_content: bytes) -> str:
        try:
            msg = BytesParser(policy=default).parsebytes(file_content)
            text = ""
            if msg['Subject']:
                text += f"Subject: {msg['Subject']}\n\n"
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            text += payload.decode('utf-8', errors='ignore')
                    elif part.get_content_type() == "text/html" and BS4_AVAILABLE:
                        payload = part.get_payload(decode=True)
                        if payload:
                            soup = BeautifulSoup(payload, 'html.parser')
                            text += soup.get_text()
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    text += payload.decode('utf-8', errors='ignore')
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting EML text: {e}")

    def _download_and_extract_text(self, url: str) -> str:
        ftype = self._detect_file_type(url)
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            raise Exception(f"Failed to download file: {resp.status_code}")
        content = resp.content
        if ftype == 'pdf':
            return self._extract_text_from_pdf(content)
        elif ftype == 'docx':
            return self._extract_text_from_docx(content)
        elif ftype == 'eml':
            return self._extract_text_from_eml(content)
        try:
            return self._extract_text_from_pdf(content)
        except:
            return content.decode('utf-8', errors='ignore')

    def process_document_from_url(self, doc_url: str, questions: List[str]) -> List[str]:
        document_text = self._download_and_extract_text(doc_url)
        if not document_text.strip():
            raise Exception("No text extracted")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            length_function=len
        )
        docs = [Document(page_content=document_text)]
        chunks = splitter.split_documents(docs)

        embeddings = self._get_embeddings()
        texts = [c.page_content for c in chunks]
        if FAISS_AVAILABLE:
            vectors = embeddings.embed_documents(texts)
            dim = len(vectors[0])
            index = faiss.IndexFlatL2(dim)
            index.add(np.array(vectors, dtype=np.float32))
            self.chunk_texts = texts
            self.faiss_index = index
        else:
            self.chunk_texts = texts

        llm = self._get_llm()
        parser = StrOutputParser()
        answers = []
        for q in questions:
            if FAISS_AVAILABLE:
                context = self._get_relevant_context_faiss(q, k=3)
            else:
                context = self._get_relevant_context_simple(q, k=3)
            prompt = self.prompt_template.format(context=context, question=q)
            response = llm.invoke(prompt)
            answers.append(parser.invoke(response).strip())
        return answers

    def _get_relevant_context_faiss(self, question: str, k=3) -> str:
        qvec = self.embeddings.embed_query(question)
        distances, indices = self.faiss_index.search(
            np.array([qvec], dtype=np.float32), k
        )
        return "\n\n".join(self.chunk_texts[idx] for idx in indices[0] if idx < len(self.chunk_texts))

    def _get_relevant_context_simple(self, question: str, k=3) -> str:
        q_words = set(question.lower().split())
        scored = [(len(q_words & set(c.lower().split())), i, c) for i, c in enumerate(self.chunk_texts)]
        scored.sort(reverse=True, key=lambda x: x[0])
        return "\n\n".join(c for _, _, c in scored[:k]) or "\n\n".join(self.chunk_texts[:k])
