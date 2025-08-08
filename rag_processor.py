import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader
from langchain_community.vectorstores import FAISS

load_dotenv()

# Read Gemini API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set in environment variables.")

class RAGProcessor:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_API_KEY  # ✅ Explicitly set API key
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            google_api_key=GEMINI_API_KEY  # ✅ Explicitly set API key
        )

    def load_document(self, file_path):
        """Loads document depending on type"""
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".eml"):
            loader = UnstructuredEmailLoader(file_path)
        else:
            raise ValueError("Unsupported file type. Supported: PDF, DOCX, EML")
        return loader.load()

    def process(self, file_path):
        """Splits, embeds, and stores document in FAISS"""
        documents = self.load_document(file_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(documents)

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

    def query(self, question):
        """Retrieves answer from vector store"""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(question)

        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"Answer the following question using the provided context:\n\nContext:\n{context}\n\nQuestion: {question}"

        response = self.llm.invoke(prompt)
        return response.content
