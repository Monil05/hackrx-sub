import os
import requests
import gc
from typing import List
from email.parser import BytesParser
from email.policy import default
import io
import numpy as np

# Try to import PDF/DOCX/HTML helpers if available
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

# LangChain-ish utilities (lightweight split + LLM wrapper)
try:
    # langchain-text-splitters provides this module name
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    # fallback import name if package differs
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# Google Generative AI LangChain adapter (from langchain-google-genai)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    # If the package exposes different name, this will let code fail later clearly
    raise

# Output parser and Document container compatible with langchain-core package
try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.documents import Document
except Exception:
    # If langchain_core is not present, raise a clear error early
    raise

# Optional FAISS + sentence-transformers for better retrieval
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False


class SimpleEmbeddings:
    """Lightweight embedding wrapper that uses sentence-transformers when available."""
    def __init__(self):
        if FAISS_AVAILABLE:
            # use a compact model
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            self.model = None

    def embed_documents(self, texts: List[str]):
        if self.model:
            vecs = self.model.encode(texts, convert_to_tensor=False)
            # ensure list of lists
            return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]
        # fallback: zero vectors (dim 384)
        return [[0.0] * 384 for _ in texts]

    def embed_query(self, text: str):
        if self.model:
            v = self.model.encode([text], convert_to_tensor=False)[0]
            return v.tolist() if hasattr(v, "tolist") else list(v)
        return [0.0] * 384


class RAGProcessor:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY (or equivalent) must be provided in environment")
        self.api_key = api_key
        self.llm = None
        self.embeddings = None
        self.faiss_index = None
        self.chunk_texts = []

        # Simple prompt template
        self.prompt_template = (
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. Keep the answer concise and accurate.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

    def _get_llm(self):
        """Lazy-load LLM wrapper to reduce memory at startup"""
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
        url_lower = (url or "").lower()
        if ".pdf" in url_lower or "application/pdf" in url_lower:
            return "pdf"
        if ".docx" in url_lower or "word" in url_lower:
            return "docx"
        if ".eml" in url_lower or "message/rfc822" in url_lower:
            return "eml"
        # HEAD request as fallback
        try:
            resp = requests.head(url, timeout=5)
            ctype = resp.headers.get("content-type", "").lower()
            if "pdf" in ctype:
                return "pdf"
            if "word" in ctype or "officedocument" in ctype:
                return "docx"
            if "message/rfc822" in ctype or "eml" in ctype:
                return "eml"
        except Exception:
            pass
        return "pdf"

    def _extract_text_from_pdf(self, file_content: bytes) -> str:
        if not PDF_AVAILABLE:
            raise Exception("PyPDF2 is required for PDF processing but not installed.")
        try:
            pdf_file = io.BytesIO(file_content)
            reader = PyPDF2.PdfReader(pdf_file)
            parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    parts.append(text)
            return "\n".join(parts).strip()
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {e}")

    def _extract_text_from_docx(self, file_content: bytes) -> str:
        if not DOCX_AVAILABLE:
            raise Exception("python-docx is required for DOCX processing but not installed.")
        try:
            docx_io = io.BytesIO(file_content)
            doc = DocxDocument(docx_io)
            return "\n".join(p.text for p in doc.paragraphs).strip()
        except Exception as e:
            raise Exception(f"Error extracting DOCX text: {e}")

    def _extract_text_from_eml(self, file_content: bytes) -> str:
        try:
            msg = BytesParser(policy=default).parsebytes(file_content)
            text = ""
            if msg["Subject"]:
                text += f"Subject: {msg['Subject']}\n\n"
            if msg.is_multipart():
                for part in msg.walk():
                    ctype = part.get_content_type()
                    if ctype == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            text += payload.decode("utf-8", errors="ignore")
                    elif ctype == "text/html" and BS4_AVAILABLE:
                        payload = part.get_payload(decode=True)
                        if payload:
                            soup = BeautifulSoup(payload, "html.parser")
                            text += soup.get_text()
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    text += payload.decode("utf-8", errors="ignore")
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting EML text: {e}")

    def _download_and_extract_text(self, url: str) -> str:
        ftype = self._detect_file_type(url)
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            raise Exception(f"Failed to download file: {resp.status_code}")
        content = resp.content
        if ftype == "pdf":
            return self._extract_text_from_pdf(content)
        if ftype == "docx":
            return self._extract_text_from_docx(content)
        if ftype == "eml":
            return self._extract_text_from_eml(content)
        # fallback try PDF then plain text
        try:
            return self._extract_text_from_pdf(content)
        except Exception:
            return content.decode("utf-8", errors="ignore")

    def process_document_from_url(self, doc_url: str, questions: List[str]) -> List[str]:
        try:
            # Step 1: download & extract
            document_text = self._download_and_extract_text(doc_url)
            if not document_text or not document_text.strip():
                raise Exception("No text could be extracted from the document.")

            # Step 2: split into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50, length_function=len)
            docs = [Document(page_content=document_text)]
            chunks = splitter.split_documents(docs)
            if not chunks:
                raise Exception("Document could not be split into chunks.")

            # Clear original text
            del docs, document_text
            gc.collect()

            # Step 3: embeddings / FAISS index if available
            emb = self._get_embeddings()
            texts = [c.page_content for c in chunks]

            if FAISS_AVAILABLE:
                vectors = emb.embed_documents(texts)
                dim = len(vectors[0])
                index = faiss.IndexFlatL2(dim)
                index.add(np.array(vectors, dtype=np.float32))
                self.faiss_index = index
                self.chunk_texts = texts
            else:
                # simple fallback store
                self.chunk_texts = texts

            # Free chunk objects
            del chunks, texts
            gc.collect()

            # Step 4: answer questions
            llm = self._get_llm()
            parser = StrOutputParser()
            answers = []

            for q in questions:
                try:
                    if FAISS_AVAILABLE:
                        context = self._get_relevant_context_faiss(q, k=3)
                    else:
                        context = self._get_relevant_context_simple(q, k=3)

                    prompt = self.prompt_template.format(context=context, question=q)
                    # Use LLM invoke/predict depending on wrapper API
                    # ChatGoogleGenerativeAI typically provides .invoke or .generate — use .invoke() here
                    response = llm.invoke(prompt)
                    # Parse/cleanup
                    ans = parser.invoke(response) if parser else (response if isinstance(response, str) else str(response))
                    answers.append(ans.strip())
                    gc.collect()
                except Exception as e:
                    answers.append(f"Error processing question: {str(e)}")

            return answers

        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")

    def _get_relevant_context_faiss(self, question: str, k: int = 3) -> str:
        try:
            qvec = self.embeddings.embed_query(question)
            distances, indices = self.faiss_index.search(np.array([qvec], dtype=np.float32), k)
            relevant = [self.chunk_texts[i] for i in indices[0] if i < len(self.chunk_texts)]
            return "\n\n".join(relevant)
        except Exception:
            return "\n\n".join(self.chunk_texts[:k])

    def _get_relevant_context_simple(self, question: str, k: int = 3) -> str:
        qwords = set(question.lower().split())
        scored = []
        for i, txt in enumerate(self.chunk_texts):
            overlap = len(qwords.intersection(set(txt.lower().split())))
            scored.append((overlap, i, txt))
        scored.sort(reverse=True, key=lambda x: x[0])
        rel = [t for _, _, t in scored[:k]]
        if not rel:
            rel = self.chunk_texts[:k]
        return "\n\n".join(rel)
