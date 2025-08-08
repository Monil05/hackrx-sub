import os
import tempfile
import requests
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

class RAGProcessor:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
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
        return text

    def split_text(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
        docs = splitter.split_text(text)
        return [Document(page_content=chunk) for chunk in docs]

    def create_vectorstore(self, documents):
        return FAISS.from_documents(documents, self.embeddings)

    def run_rag(self, doc_url, questions):
        # Step 1: Download and load document
        text = self.load_pdf_from_url(doc_url)

        # Step 2: Split into chunks
        docs = self.split_text(text)

        # Step 3: Embed + store
        vectorstore = self.create_vectorstore(docs)

        # Step 4: Retrieve and answer
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        answers = []
        for q in questions:
            rel_docs = retriever.get_relevant_documents(q)
            context = "\n\n".join([d.page_content for d in rel_docs])
            prompt = f"Answer the question based on the document:\n\n{context}\n\nQuestion: {q}"
            res = self.llm.predict(prompt)
            answers.append(res.strip())

        return answers
