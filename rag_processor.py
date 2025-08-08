import requests
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS


class RAGProcessor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, google_api_key=api_key)

    def process(self, document_url, questions):
        # Download document
        temp_file = self._download_document(document_url)

        # Load document
        loader = PyPDFLoader(temp_file)
        docs = loader.load()

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Embed & store
        vectorstore = FAISS.from_documents(splits, self.embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # Generate answers
        answers = []
        for question in questions:
            context_docs = retriever.get_relevant_documents(question)
            context = "\n\n".join(doc.page_content for doc in context_docs)
            prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
            answer = self.llm.predict(prompt)
            answers.append(answer.strip())

        # Clean up
        os.remove(temp_file)
        return answers

    def _download_document(self, url):
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Failed to download document: {response.status_code}")

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name
