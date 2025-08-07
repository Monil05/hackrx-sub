import os
import tempfile
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI  # or HuggingFace if using Mixtral
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub


class RAGProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.prompt = hub.pull("rlm/rag-prompt")
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def process_pdf_from_url(self, pdf_url: str, questions: list[str]) -> list[str]:
        # === Step 1: Download PDF to temp file ===
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            response = requests.get(pdf_url)
            if response.status_code != 200:
                raise Exception(f"Failed to download PDF: {response.status_code}")
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        # === Step 2: Load PDF from temp file ===
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        # === Step 3: Split into chunks ===
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        pages = splitter.split_documents(docs)

        # === Step 4: Embed & store ===
        vector_store = InMemoryVectorStore.from_documents(pages, self.embeddings)
        retriever = vector_store.as_retriever()

        # === Step 5: RAG pipeline ===
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # === Step 6: Ask each question ===
        answers = []
        for q in questions:
            try:
                result = rag_chain.invoke(q)
                answers.append(result)
            except Exception as e:
                answers.append(f"Error answering question: {str(e)}")

        return answers
