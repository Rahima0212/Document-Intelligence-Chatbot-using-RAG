# Streamlit RAG app (ChromaDB + Gemini)

This repository contains a Streamlit-based Retrieval-Augmented Generation (RAG) application using ChromaDB for vector storage and the Gemini API for LLM/embeddings.

Features
- Upload documents (PDF, DOCX, TXT, images)
- OCR for scanned/image-based docs
- Chunking and embedding storage in ChromaDB
- Chat UI with context preservation and "New Chat"

Setup
1. Create a virtual environment and install requirements:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Install Tesseract OCR on your system and ensure `tesseract` is in PATH.

3. Copy `.env.example` to `.env` and populate keys for GEMINI_API_KEY and GEMINI_API_URL.

4. Run the Streamlit app:

```powershell
streamlit run app.py
```

Notes
- This code expects a Gemini-compatible HTTP API for embeddings and completions. Provide your endpoint and key in `.env`.
- If you don't have Gemini embeddings available, the code falls back to a local Sentence-Transformers model for embeddings.
