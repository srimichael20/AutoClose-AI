"""SQLite database for documents and transactions."""

from pathlib import Path
from typing import Any

import aiosqlite

from utils.config import get_settings
from utils.schemas import ClassificationResult, DocumentType


class Database:
    """Async SQLite persistence."""

    def __init__(self, db_path: str | None = None) -> None:
        settings = get_settings()
        self.path = db_path or settings.database_path
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init = False

    async def _ensure_schema(self, conn: aiosqlite.Connection) -> None:
        if self._init:
            return
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, document_type TEXT, raw_content TEXT, file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, document_id TEXT, category TEXT,
                subcategory TEXT, amount REAL, description TEXT, confidence REAL, reasoning TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            CREATE INDEX IF NOT EXISTS idx_tx_doc ON transactions(document_id);
        """)
        await conn.commit()
        self._init = True

    async def _conn(self) -> aiosqlite.Connection:
        conn = await aiosqlite.connect(self.path)
        await self._ensure_schema(conn)
        return conn

    async def store_document(
        self,
        document_id: str,
        document_type: DocumentType,
        raw_content: str | None = None,
        file_path: str | None = None,
    ) -> bool:
        try:
            conn = await self._conn()
            await conn.execute(
                "INSERT OR REPLACE INTO documents (id, document_type, raw_content, file_path) VALUES (?,?,?,?)",
                (document_id, document_type.value, raw_content, file_path),
            )
            await conn.commit()
            await conn.close()
            return True
        except Exception:
            return False

    async def store_transaction(self, document_id: str, classification: ClassificationResult) -> bool:
        try:
            conn = await self._conn()
            await conn.execute(
                """INSERT INTO transactions (document_id, category, subcategory, amount, description, confidence, reasoning)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    document_id,
                    classification.category.value,
                    classification.subcategory,
                    classification.amount,
                    classification.description,
                    classification.confidence,
                    classification.reasoning,
                ),
            )
            await conn.commit()
            await conn.close()
            return True
        except Exception:
            return False
