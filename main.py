from flask import Flask, request, jsonify
import os
from rag_processor import RAGProcessor
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
processor = RAGProcessor()

AUTH_TOKEN = os.getenv("AUTH_TOKEN")

@app.route("/hackrx/run", methods=["POST"])
def hackrx_run():
    try:
        # Check Bearer token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        token = auth_header.split("Bearer ")[1].strip()
        if token != AUTH_TOKEN:
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json()
        doc_url = data.get("documents")
        questions = data.get("questions")

        if not doc_url or not questions:
            return jsonify({"error": "Missing 'documents' or 'questions'"}), 400

        # Process with RAG
        answers = processor.run_rag(doc_url, questions)

        return jsonify({"answers": answers}), 200

    except Exception as e:
        # Log error for debugging in Render logs
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)