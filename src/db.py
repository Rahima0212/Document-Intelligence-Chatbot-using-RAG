import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

class LocalStorage:
    def __init__(self, db_path: str = "local_storage.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    filename TEXT NOT NULL,
                    metadata TEXT,  -- JSON string for additional info
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)

    def create_conversation(self, conversation_id: str, title: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO conversations (id, title) VALUES (?, ?)",
                (conversation_id, title)
            )

    def update_conversation_title(self, conversation_id: str, new_title: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, last_updated = CURRENT_TIMESTAMP WHERE id = ?",
                (new_title, conversation_id)
            )

    def add_message(self, conversation_id: str, message_id: str, role: str, content: str, metadata: Dict[str, Any] = None) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, metadata) VALUES (?, ?, ?, ?, ?)",
                (message_id, conversation_id, role, content, json.dumps(metadata) if metadata else None)
            )
            conn.execute(
                "UPDATE conversations SET last_updated = CURRENT_TIMESTAMP WHERE id = ?",
                (conversation_id,)
            )

    def add_document(self, conversation_id: str, document_id: str, filename: str, metadata: Dict[str, Any] = None) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO documents (id, conversation_id, filename, metadata) VALUES (?, ?, ?, ?)",
                (document_id, conversation_id, filename, json.dumps(metadata) if metadata else None)
            )

    def get_conversations(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM conversations ORDER BY last_updated DESC"
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at",
                (conversation_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_conversation_documents(self, conversation_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM documents WHERE conversation_id = ? ORDER BY created_at",
                (conversation_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def delete_conversation(self, conversation_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM documents WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))