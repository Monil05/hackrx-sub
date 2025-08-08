import os
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredEmailLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class RAGProcessor:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY, temperature=0)
        self.vector_store = None

    def load_document(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = UnstructuredWordDocumentLoader(file_path)
        elif ext == ".eml":
            loader = UnstructuredEmailLoader(file_path)
        else:
            raise ValueError("Unsupported file format")
        return loader.load()

    def split_and_store(self, docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        self.vector_store = InMemoryVectorStore.from_documents(chunks, self.embeddings)

    def run_rag(self, question):
        if not self.vector_store:
            raise ValueError("No documents loaded")
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = f"Answer the question based on the context:\n\n{context}\n\nQuestion: {question}"
        return self.llm.invoke(prompt).content
