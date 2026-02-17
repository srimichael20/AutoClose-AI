"""File storage for uploads and processed results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles

from utils.config import get_settings
from utils.schemas import ClassificationResult, DocumentType, VisionResult


class FileStorage:
    def __init__(
        self,
        upload_dir: str | None = None,
        processed_dir: str | None = None,
    ) -> None:
        s = get_settings()
        self.upload = Path(upload_dir or s.upload_directory)
        self.processed = Path(processed_dir or s.processed_directory)
        self.upload.mkdir(parents=True, exist_ok=True)
        self.processed.mkdir(parents=True, exist_ok=True)

    def store_upload_sync(
        self,
        document_id: str,
        content: bytes,
        document_type: DocumentType,
    ) -> str:
        """Synchronous upload for Streamlit (avoids asyncio conflicts)."""
        ext = {DocumentType.PDF: ".pdf", DocumentType.IMAGE: ".png", DocumentType.TEXT: ".txt"}.get(
            document_type, ".bin"
        )
        path = self.upload / f"{document_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}{ext}"
        path.write_bytes(content)
        return str(path)

    async def store_upload(
        self,
        document_id: str,
        content: bytes,
        document_type: DocumentType,
    ) -> str:
        ext = {DocumentType.PDF: ".pdf", DocumentType.IMAGE: ".png", DocumentType.TEXT: ".txt"}.get(
            document_type, ".bin"
        )
        path = self.upload / f"{document_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}{ext}"
        async with aiofiles.open(path, "wb") as f:
            await f.write(content)
        return str(path)

    async def store_result(
        self,
        document_id: str,
        vision: VisionResult | None = None,
        classification: ClassificationResult | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        data = {
            "document_id": document_id,
            "processed_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        if vision:
            data["vision"] = vision.model_dump()
        if classification:
            data["classification"] = classification.model_dump()
        path = self.processed / f"{document_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, indent=2, default=str))
        return str(path)
