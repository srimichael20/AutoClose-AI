"""MCP Database service - SQLite for accounting records."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from autoclose.config import get_settings
from autoclose.schemas import ClassificationResult, DocumentType


class DatabaseService:
    """Async SQLite database service for storing processed accounting records."""

    def __init__(self, db_path: str | None = None) -> None:
        settings = get_settings()
        self.db_path = db_path or settings.database_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    async def _ensure_schema(self, conn: aiosqlite.Connection) -> None:
        """Create tables if they don't exist."""
        await conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                document_type TEXT NOT NULL,
                raw_content TEXT,
                file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                category TEXT NOT NULL,
                subcategory TEXT,
                amount REAL,
                description TEXT,
                confidence REAL,
                reasoning TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            );

            CREATE INDEX IF NOT EXISTS idx_transactions_document_id
                ON transactions(document_id);
            CREATE INDEX IF NOT EXISTS idx_transactions_category
                ON transactions(category);
        """)
        await conn.commit()

    async def _get_connection(self) -> aiosqlite.Connection:
        """Get database connection with schema initialized."""
        conn = await aiosqlite.connect(self.db_path)
        if not self._initialized:
            await self._ensure_schema(conn)
            self._initialized = True
        return conn

    async def store_document(
        self,
        document_id: str,
        document_type: DocumentType,
        raw_content: str | None = None,
        file_path: str | None = None,
    ) -> bool:
        """Store document metadata in the database."""
        try:
            conn = await self._get_connection()
            await conn.execute(
                """
                INSERT OR REPLACE INTO documents (id, document_type, raw_content, file_path)
                VALUES (?, ?, ?, ?)
                """,
                (document_id, document_type.value, raw_content, file_path),
            )
            await conn.commit()
            await conn.close()
            return True
        except Exception:
            return False

    async def store_transaction(
        self,
        document_id: str,
        classification: ClassificationResult,
    ) -> bool:
        """Store classified transaction in the database."""
        try:
            conn = await self._get_connection()
            await conn.execute(
                """
                INSERT INTO transactions
                (document_id, category, subcategory, amount, description, confidence, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
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

    async def get_transactions(
        self,
        document_id: str | None = None,
        category: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve transactions with optional filters."""
        conn = await self._get_connection()
        query = "SELECT * FROM transactions WHERE 1=1"
        params: list[Any] = []

        if document_id:
            query += " AND document_id = ?"
            params.append(document_id)
        if category:
            query += " AND category = ?"
            params.append(category)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()
        columns = [d[0] for d in cursor.description]
        await conn.close()

        return [dict(zip(columns, row)) for row in rows]
