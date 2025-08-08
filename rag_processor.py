import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import os
import tempfile

class RAGProcessor:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0
        )

    def _download_pdf(self, url):
        """Downloads a PDF from a given URL and returns the local file path."""
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "").lower()
            if "pdf" not in content_type:
                raise ValueError(f"Invalid file type: {content_type}")

            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp_file.write(resp.content)
            tmp_file.close()
            return tmp_file.name
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error downloading PDF: {str(e)}")
        except Exception as e:
            raise ValueError(f"Invalid PDF URL or file: {str(e)}")

    def process(self, document_url, questions):
        """Processes the document URL and answers questions."""
        try:
            # Step 1: Download and verify PDF
            pdf_path = self._download_pdf(document_url)

            # Step 2: Load and split PDF
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = splitter.split_documents(docs)

            # Step 3: Embed and store in FAISS
            vectorstore = FAISS.from_documents(chunks, self.embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

            # Step 4: Generate answers
            answers = []
            for q in questions:
                context = retriever.get_relevant_documents(q)
                context_text = "\n\n".join([c.page_content for c in context])
                prompt = f"Context:\n{context_text}\n\nQuestion: {q}\nAnswer:"
                ans = self.llm.invoke(prompt)
                answers.append(ans.content.strip())

            return {"answers": answers}

        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
