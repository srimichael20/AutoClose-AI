"""Pydantic schemas for workflow state and data transfer."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported document types for intake."""

    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"


class TransactionCategory(str, Enum):
    """Accounting transaction categories for classification."""

    REVENUE = "revenue"
    EXPENSE = "expense"
    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Workflow processing status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class IntakeResult(BaseModel):
    """Output from Intake Agent."""

    document_id: str
    document_type: DocumentType
    raw_content: str | None = None
    file_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    requires_vision: bool = False


class VisionResult(BaseModel):
    """Output from Vision Agent (OCR + extraction)."""

    document_id: str
    extracted_text: str
    structured_data: dict[str, Any] = Field(default_factory=dict)
    confidence_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClassificationResult(BaseModel):
    """Output from Classification Agent."""

    document_id: str
    category: TransactionCategory
    subcategory: str | None = None
    amount: float | None = None
    description: str | None = None
    confidence: float = 0.0
    reasoning: str | None = None
    embeddings_added: bool = False


class MCPActionResult(BaseModel):
    """Output from MCP Integration Agent."""

    document_id: str
    database_recorded: bool = False
    file_stored: bool = False
    api_called: bool = False
    notification_sent: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class WorkflowState(BaseModel):
    """Shared state passed through the agent workflow graph."""

    # Input
    job_id: str
    document_id: str
    file_path: str | None = None
    raw_content: str | None = None
    document_type: DocumentType = DocumentType.TEXT

    # Agent outputs
    intake_result: IntakeResult | None = None
    vision_result: VisionResult | None = None
    classification_result: ClassificationResult | None = None
    mcp_result: MCPActionResult | None = None

    # Control flow
    current_step: str = "intake"
    status: ProcessingStatus = ProcessingStatus.PENDING
    error: str | None = None
    messages: list[str] = Field(default_factory=list)

    # For LLM context
    context_for_next_agent: str = ""

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class WorkflowRequest(BaseModel):
    """API request for workflow submission."""

    document_id: str | None = None
    document_type: DocumentType = DocumentType.TEXT
    content: str | None = None
    file_path: str | None = None


class WorkflowResponse(BaseModel):
    """API response for workflow completion."""

    job_id: str
    document_id: str
    status: ProcessingStatus
    intake_result: IntakeResult | None = None
    vision_result: VisionResult | None = None
    classification_result: ClassificationResult | None = None
    mcp_result: MCPActionResult | None = None
    error: str | None = None
