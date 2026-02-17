"""Pydantic schemas for workflow state and API."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"


class TransactionCategory(str, Enum):
    REVENUE = "revenue"
    EXPENSE = "expense"
    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class IntakeResult(BaseModel):
    document_id: str
    document_type: DocumentType
    raw_content: str | None = None
    file_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    requires_vision: bool = False


class VisionResult(BaseModel):
    document_id: str
    extracted_text: str
    structured_data: dict[str, Any] = Field(default_factory=dict)
    confidence_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClassificationResult(BaseModel):
    document_id: str
    category: TransactionCategory
    subcategory: str | None = None
    amount: float | None = None
    description: str | None = None
    confidence: float = 0.0
    reasoning: str | None = None
    embeddings_added: bool = False


class MCPActionResult(BaseModel):
    document_id: str
    database_recorded: bool = False
    file_stored: bool = False
    api_called: bool = False
    notification_sent: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class SummaryResult(BaseModel):
    document_id: str
    financial_summary: str
    user_prompt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowState(BaseModel):
    job_id: str
    document_id: str
    file_path: str | None = None
    raw_content: str | None = None
    document_type: DocumentType = DocumentType.TEXT
    user_prompt: str | None = None

    intake_result: IntakeResult | None = None
    vision_result: VisionResult | None = None
    classification_result: ClassificationResult | None = None
    mcp_result: MCPActionResult | None = None
    summary_result: SummaryResult | None = None

    current_step: str = "intake"
    status: ProcessingStatus = ProcessingStatus.PENDING
    error: str | None = None
    messages: list[str] = Field(default_factory=list)
    context_for_next_agent: str = ""

    class Config:
        arbitrary_types_allowed = True


class WorkflowRequest(BaseModel):
    document_id: str | None = None
    document_type: DocumentType = DocumentType.TEXT
    content: str | None = None
    file_path: str | None = None
    user_prompt: str | None = None


class WorkflowResponse(BaseModel):
    job_id: str
    document_id: str
    status: ProcessingStatus
    intake_result: IntakeResult | None = None
    vision_result: VisionResult | None = None
    classification_result: ClassificationResult | None = None
    mcp_result: MCPActionResult | None = None
    summary_result: SummaryResult | None = None
    error: str | None = None
