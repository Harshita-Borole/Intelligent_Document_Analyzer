import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ---------------- GEMINI IMPORT ----------------
try:
    from google import genai
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

# ---------------- CONFIG ----------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None
    GEMINI_AVAILABLE = False

# ---------------- FLASK SETUP ----------------
app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- GLOBAL STORAGE ----------------
DOCUMENT_CHUNKS = []
FAISS_INDEX = None
GEMINI_CACHE = {}

embedder = SentenceTransformer("all-MiniLM-L6-v2")

MAX_QA_CHUNKS = 6
MAX_SUMMARY_CHUNKS = 15   # covers almost full document

# ---------------- HELPERS ----------------
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return " ".join(page.get_text() for page in doc).strip()

def chunk_text(text, max_chars=900):
    words = text.split()
    chunks, current = [], ""
    for w in words:
        if len(current) + len(w) < max_chars:
            current += " " + w
        else:
            chunks.append(current.strip())
            current = w
    if current:
        chunks.append(current.strip())
    return chunks

def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index

# ---------------- GEMINI CALL ----------------
def gemini_call(prompt):
    if not GEMINI_AVAILABLE:
        print("❌ Gemini not configured")
        return None, False

    if prompt in GEMINI_CACHE:
        return GEMINI_CACHE[prompt], True

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        text = response.text
        GEMINI_CACHE[prompt] = text
        print("✅ Gemini used")
        return text, True

    except Exception as e:
        print("❌ Gemini failed:", e)
        return None, False

# ---------------- FALLBACK ----------------
def fallback_summary(chunks):
    text = " ".join(chunks)
    sentences = text.split(".")
    return ". ".join(sentences[:20]).strip() + "."

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return jsonify({"status": "Intelligent Document Assistant backend running"})

# ---------------- UPLOAD ----------------
@app.route("/upload", methods=["POST"])
def upload_pdf():
    global DOCUMENT_CHUNKS, FAISS_INDEX

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(path)

    text = extract_text_from_pdf(path)
    DOCUMENT_CHUNKS = chunk_text(text)
    FAISS_INDEX = build_faiss_index(DOCUMENT_CHUNKS)

    return jsonify({
        "message": "PDF indexed successfully",
        "characters": len(text),
        "chunks": len(DOCUMENT_CHUNKS)
    })

# ---------------- QUESTION ANSWERING ----------------
@app.route("/ask", methods=["POST"])
def ask_question():
    if not DOCUMENT_CHUNKS:
        return jsonify({"error": "Upload a document first"}), 400

    data = request.get_json()
    question = data.get("question", "").strip()

    q_emb = embedder.encode([question])
    _, idx = FAISS_INDEX.search(np.array(q_emb), MAX_QA_CHUNKS)
    context = "\n".join(DOCUMENT_CHUNKS[i] for i in idx[0])

    prompt = f"""
You are an expert academic assistant.

Answer the question using ONLY the information provided below.
Explain in clear, detailed paragraphs.

Requirements:
- Define important terms
- Explain algorithms or procedures step by step if present
- Explain formulas if mentioned
- Use exam-oriented language
- Write like a 5–10 mark university answer

Document Content:
{context}

Question:
{question}

Final Answer:
"""

    answer, used = gemini_call(prompt)
    if not used:
        answer = fallback_summary([context])

    return jsonify({
        "answer": answer,
        "gemini_used": used
    })

# ---------------- FULL DOCUMENT SUMMARY ----------------
@app.route("/summarize", methods=["POST"])
def summarize_document():
    if not DOCUMENT_CHUNKS:
        return jsonify({"error": "Upload a document first"}), 400

    query = "Complete explanation of all concepts, algorithms, and topics"
    q_emb = embedder.encode([query])
    _, idx = FAISS_INDEX.search(np.array(q_emb), MAX_SUMMARY_CHUNKS)
    selected_chunks = [DOCUMENT_CHUNKS[i] for i in idx[0]]

    merged_text = "\n".join(selected_chunks)

    prompt = f"""
You are an expert academic reader and content analyst.

Create a COMPLETE and COMPREHENSIVE summary of the provided document.

STRICT INSTRUCTIONS:
- Do NOT skip any important topic or concept
- Explain all algorithms, procedures, and steps clearly
- Explain formulas and their meaning if present
- Include advantages, disadvantages, and limitations if mentioned
- Maintain the logical flow of the original document
- Avoid over-compression; clarity is more important than brevity

STYLE REQUIREMENTS:
- Paragraph format only (NO bullet points)
- Simple, clear academic language
- Suitable for full exam revision
- Explain as if teaching a student

Document Content:
{merged_text}

Final Detailed Summary:
"""

    summary, used = gemini_call(prompt)
    if not used:
        summary = fallback_summary(selected_chunks)

    return jsonify({
        "summary": summary,
        "gemini_used": used
    })

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(port=5000, debug=False)
