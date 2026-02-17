"""MCP File System service - document storage and retrieval."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles

from autoclose.config import get_settings
from autoclose.schemas import ClassificationResult, DocumentType, VisionResult


class FileSystemService:
    """Async file system service for document storage and archiving."""

    def __init__(
        self,
        upload_directory: str | None = None,
        processed_directory: str | None = None,
    ) -> None:
        settings = get_settings()
        self.upload_dir = Path(upload_directory or settings.upload_directory)
        self.processed_dir = Path(processed_directory or settings.processed_directory)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _generate_filename(self, document_id: str, suffix: str = "") -> str:
        """Generate unique filename for storage."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{document_id}_{ts}{suffix}"

    async def store_upload(
        self,
        document_id: str,
        content: bytes,
        document_type: DocumentType,
        original_filename: str | None = None,
    ) -> str:
        """Store uploaded file and return path."""
        ext_map = {
            DocumentType.PDF: ".pdf",
            DocumentType.IMAGE: ".png",
            DocumentType.TEXT: ".txt",
        }
        ext = ext_map.get(document_type, ".bin")
        filename = self._generate_filename(document_id, ext)
        file_path = self.upload_dir / filename

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        return str(file_path)

    async def store_processed_result(
        self,
        document_id: str,
        vision_result: VisionResult | None = None,
        classification_result: ClassificationResult | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store processed workflow result as JSON for audit trail."""
        data: dict[str, Any] = {
            "document_id": document_id,
            "processed_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        if vision_result:
            data["vision"] = vision_result.model_dump()
        if classification_result:
            data["classification"] = classification_result.model_dump()

        filename = self._generate_filename(document_id, ".json")
        file_path = self.processed_dir / filename

        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, indent=2, default=str))

        return str(file_path)

    async def read_file(self, file_path: str) -> bytes:
        """Read file contents."""
        async with aiofiles.open(file_path, "rb") as f:
            return await f.read()

    async def read_text_file(self, file_path: str) -> str:
        """Read text file contents."""
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            return await f.read()
