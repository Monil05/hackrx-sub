import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from rag_processor import RAGProcessor

# Load environment variables
load_dotenv()

AUTH_TOKEN = os.getenv("AUTH_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__)

# Initialize RAG processor
rag = RAGProcessor(GEMINI_API_KEY)


@app.route("/hackrx/run", methods=["POST"])
def run_rag():
    # Authentication
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer ") or auth_header.split(" ")[1] != AUTH_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()

    if not data or "documents" not in data or "questions" not in data:
        return jsonify({"error": "Invalid request format"}), 400

    document_url = data["documents"]
    questions = data["questions"]

    try:
        answers = rag.process(document_url, questions)
        return jsonify({"answers": answers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
