from flask import Flask, request, jsonify
from rag_processor import RAGProcessor
import os

app = Flask(__name__)

# Load your Gemini or Mixtral key from environment (already on Render later)
API_KEY = os.getenv("GEMINI_API_KEY")  # or MIXTRAL_API_KEY
AUTH_TOKEN = os.getenv("AUTH_TOKEN")   # Token required in header

processor = RAGProcessor(API_KEY)

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
        answers = processor.process_pdf_from_url(doc_url, questions)
        return jsonify({"answers": answers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Health check route (optional) ===
@app.route("/", methods=["GET"])
def home():
    return "HackRx Webhook is live."

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # fallback for local testing
    app.run(host="0.0.0.0", port=port)
