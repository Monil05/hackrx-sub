import os
from flask import Flask, request, jsonify
from rag_processor import RAGProcessor

app = Flask(__name__)

# Load your Gemini API key and auth token from environment (Render will provide these)
API_KEY = os.getenv("GEMINI_API_KEY")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")

# Lazy-init processor to avoid heavy startup memory usage
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

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    doc_url = data.get("documents")
    questions = data.get("questions")

    if not doc_url or not isinstance(questions, list):
        return jsonify({"error": "Missing document URL or questions (questions must be a list)"}), 400

    try:
        proc = get_processor()
        answers = proc.process_document_from_url(doc_url, questions)
        return jsonify({"answers": answers})
    except Exception as e:
        # Return error message (useful for debugging) but in production consider sanitizing
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def home():
    return "HackRx Webhook is live."


if __name__ == "__main__":
    # Use Render's provided PORT if present. Fallback to 5000 for local testing.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
