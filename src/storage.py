import os
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings

from .gemini_client import GeminiClient
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST_MODEL = True
except Exception:
    _HAS_ST_MODEL = False


class ChromaStore:
    def __init__(self, persist_directory: str = None, collection_name: str = 'documents'):
        persist_directory = persist_directory or os.getenv('CHROMA_PERSIST_DIR', './chroma_data')
        # Corrected: Use Settings for client initialization
        self.client = chromadb.Client(Settings(persist_directory=persist_directory, is_persistent=True))
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # Initialize Gemini client
        self.gemini = GeminiClient()

        # Always try to load a local model if available
        self._st_model = None
        if _HAS_ST_MODEL:
            try:
                # --- THIS IS THE FIX ---
                # Specify device='cpu' to avoid the meta tensor error
                self._st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                print("✅ Loaded local embedding model: all-MiniLM-L6-v2")
            except Exception as e:
                print(f"⚠️ Failed to load SentenceTransformer model: {e}")
                self._st_model = None
        else:
            print("⚠️ sentence-transformers library not available.")

        # Show which embedding method is active
        if self.gemini.available:
            print("💡 Using Gemini for embeddings")
        elif self._st_model:
            print("💡 Using local SentenceTransformer for embeddings")
        else:
            print("❌ No embedding method available!")


    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        # embed texts via Gemini (or fallback to local model)
        embeddings = None
        if self.gemini.available:
            try:
                embeddings = self.gemini.embed_texts(texts)
            except Exception:
                embeddings = None
        if embeddings is None:
            if self._st_model is None:
                raise RuntimeError('No embedding method available: configure Gemini or install sentence-transformers')
            embeddings = self._st_model.encode(texts, show_progress_bar=False).tolist()
        self.collection.add(ids=ids, metadatas=metadatas, documents=texts, embeddings=embeddings)

    def query(self, query_text: str, top_k: int = 4, where: Dict[str, Any] = None):
        """Query the Chroma collection.

        Args:
            query_text: The natural language query to embed and search for.
            top_k: Number of results to return.
            where: Optional metadata filter to pass to ChromaDB (e.g. {'conversation_id': ...}).

        Returns:
            The raw results dict from ChromaDB containing 'documents', 'metadatas', and 'distances'.
        """
        emb = None
        if self.gemini.available:
            try:
                emb = self.gemini.embed_texts([query_text])[0]
            except Exception:
                emb = None
        if emb is None:
            if self._st_model is None:
                raise RuntimeError('No embedding method available: configure Gemini or install sentence-transformers')
            emb = self._st_model.encode([query_text], show_progress_bar=False)[0].tolist()

        # Pass the optional `where` filter through to ChromaDB's query API
        query_kwargs = {
            'query_embeddings': [emb],
            'n_results': top_k,
            'include': ['documents', 'metadatas', 'distances']
        }
        if where:
            query_kwargs['where'] = where

        results = self.collection.query(**query_kwargs)
        return results