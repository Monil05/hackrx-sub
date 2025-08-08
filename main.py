from flask import Flask, request, jsonify
from rag_processor import RAGProcessor
import os

app = Flask(__name__)

# Load your Gemini API key from environment
API_KEY = os.getenv("GEMINI_API_KEY")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")

# Initialize processor as None to create it only when needed
processor = None

def get_processor():
    global processor
    if processor is None:
        processor = RAGProcessor(API_KEY)
    return processor

@app.route("/hackrx/run", methods=["POST"])
def run_rag():
    # === Authorization check ===
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer ") or auth_header.split(" ")[1] != AUTH_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401

    # === JSON input ===
    data = request.get_json()
    doc_url = data.get("documents")
    questions = data.get("questions")

    if not doc_url or not questions:
        return jsonify({"error": "Missing document URL or questions"}), 400

    try:
        # Get processor instance
        proc = get_processor()
        answers = proc.process_document_from_url(doc_url, questions)
        return jsonify({"answers": answers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Health check route ===
@app.route("/", methods=["GET"])
def home():
    return "HackRx Webhook is live."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)