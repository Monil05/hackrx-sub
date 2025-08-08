from flask import Flask, request, jsonify
from rag_processor import RAGProcessor
import tempfile
import requests
import os

app = Flask(__name__)
processor = RAGProcessor()

AUTH_TOKEN = os.getenv("AUTH_TOKEN", "changeme")

@app.route("/hackrx/run", methods=["POST"])
def hackrx_run():
    auth_header = request.headers.get("Authorization")
    if not auth_header or auth_header != f"Bearer {AUTH_TOKEN}":
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    if not data or "documents" not in data or "questions" not in data:
        return jsonify({"error": "Missing 'documents' or 'questions'"}), 400

    doc_url = data["documents"]
    questions = data["questions"]

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            r = requests.get(doc_url)
            r.raise_for_status()
            tmp_file.write(r.content)
            tmp_path = tmp_file.name

        docs = processor.load_document(tmp_path)
        processor.split_and_store(docs)

        answers = [processor.run_rag(q) for q in questions]

        os.remove(tmp_path)
        return jsonify({"answers": answers})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
