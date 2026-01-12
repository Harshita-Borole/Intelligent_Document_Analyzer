import streamlit as st
import requests

BACKEND_URL = "http://localhost:5000"

st.set_page_config(
    page_title="Intelligent Document Assistant",
    layout="wide"
)

st.title("📄 Intelligent Document Assistant")

# ---------------- FILE UPLOAD ----------------
st.header("📤 Upload PDF")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/upload",
                files={"file": uploaded_file},
                timeout=300
            )
        except Exception as e:
            st.error(f"Upload failed: {str(e)}")
            response = None

    if response and response.ok:
        data = response.json()
        st.success("PDF indexed successfully ✅")
        st.info(f"Characters: {data['characters']} | Chunks: {data['chunks']}")
    else:
        st.error(response.json().get("error", "Upload failed") if response else "Upload failed")

st.divider()

# ---------------- SUMMARY ----------------
st.header("📝 Document Summary")

if st.button("Generate Summary"):
    with st.spinner("Summarizing..."):
        try:
            response = requests.post(f"{BACKEND_URL}/summarize", timeout=300)
        except Exception as e:
            st.error(f"Summary failed: {str(e)}")
            response = None

    if response and response.ok:
        data = response.json()
        gemini_status = "✅ Gemini AI used" if data.get("gemini_used") else "⚠️ Fallback used"
        st.info(f"Status: {gemini_status}")
        st.text_area("Summary", data["summary"], height=300)
    else:
        st.error("Summary failed")

st.divider()

# ---------------- QUESTION ANSWERING ----------------
st.header("❓ Ask a Question")

question = st.text_input("Enter your question")

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": question},
                    timeout=120
                )
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
                response = None

        if response and response.ok:
            data = response.json()
            gemini_status = "✅ Gemini AI used" if data.get("gemini_used") else "⚠️ Fallback used"
            st.info(f"Status: {gemini_status}")
            st.success("Answer:")
            st.write(data["answer"])
        else:
            st.error("Failed to get answer")
