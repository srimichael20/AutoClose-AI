"""
Sync workflow runner for Streamlit with step callbacks.
Runs: Intake → Vision (optional) → Classification → MCP → Summary
Runs in a separate thread to avoid asyncio.run() conflict with Streamlit's event loop.
"""

import asyncio
import concurrent.futures
import uuid
from pathlib import Path
from typing import Any, Callable

from agents.intake_agent import IntakeAgent
from agents.vision_agent import VisionAgent
from agents.classification_agent import ClassificationAgent
from agents.mcp_agent import MCPAgent
from agents.summary_agent import SummaryAgent
from utils.schemas import (
    DocumentType,
    WorkflowState,
    IntakeResult,
    VisionResult,
    ClassificationResult,
    MCPActionResult,
    SummaryResult,
)


def run_workflow_sync(
    document_id: str,
    document_type: DocumentType,
    file_path: str | None = None,
    raw_content: str | None = None,
    user_prompt: str | None = None,
    on_step: Callable[[str, str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Run full workflow synchronously. Calls on_step(step_name, status, data) after each step.
    """
    job_id = str(uuid.uuid4())

    def _log(step: str, status: str, data: dict[str, Any]):
        if on_step:
            on_step(step, status, data)

    async def _run() -> dict[str, Any]:
        state = WorkflowState(
            job_id=job_id,
            document_id=document_id,
            file_path=file_path,
            raw_content=raw_content,
            document_type=document_type,
            user_prompt=user_prompt,
        )

        # Intake
        _log("intake", "running", {})
        intake_agent = IntakeAgent()
        up = await intake_agent.process(state)
        state = WorkflowState(**{**state.model_dump(), **up})
        _log("intake", "done", {"intake_result": state.intake_result.model_dump() if state.intake_result else None})
        if state.status.value == "failed":
            return _final_dict(state)

        # Vision (if required)
        if state.current_step == "vision":
            _log("vision", "running", {})
            vision_agent = VisionAgent()
            up = await vision_agent.process(state)
            state = WorkflowState(**{**state.model_dump(), **up})
            _log("vision", "done", {"vision_result": state.vision_result.model_dump() if state.vision_result else None})
            if state.status.value == "failed":
                return _final_dict(state)

        # Classification
        _log("classification", "running", {})
        classification_agent = ClassificationAgent()
        up = await classification_agent.process(state)
        state = WorkflowState(**{**state.model_dump(), **up})
        _log("classification", "done", {"classification_result": state.classification_result.model_dump() if state.classification_result else None})

        # MCP
        _log("mcp", "running", {})
        mcp_agent = MCPAgent()
        up = await mcp_agent.process(state)
        state = WorkflowState(**{**state.model_dump(), **up})
        _log("mcp", "done", {"mcp_result": state.mcp_result.model_dump() if state.mcp_result else None})

        # Summary
        _log("summary", "running", {})
        summary_agent = SummaryAgent()
        up = await summary_agent.process(state)
        state = WorkflowState(**{**state.model_dump(), **up})
        _log("summary", "done", {"summary_result": state.summary_result.model_dump() if state.summary_result else None})

        return _final_dict(state)

    def _final_dict(s: WorkflowState) -> dict[str, Any]:
        return {
            "job_id": s.job_id,
            "document_id": s.document_id,
            "file_path": s.file_path,
            "raw_content": s.raw_content,
            "document_type": s.document_type.value if hasattr(s.document_type, "value") else str(s.document_type),
            "user_prompt": s.user_prompt,
            "intake_result": s.intake_result.model_dump() if s.intake_result else None,
            "vision_result": s.vision_result.model_dump() if s.vision_result else None,
            "classification_result": s.classification_result.model_dump() if s.classification_result else None,
            "mcp_result": s.mcp_result.model_dump() if s.mcp_result else None,
            "summary_result": s.summary_result.model_dump() if s.summary_result else None,
            "current_step": s.current_step,
            "status": s.status.value,
            "error": s.error,
            "messages": s.messages,
            "context_for_next_agent": s.context_for_next_agent,
        }

    # Run in separate thread to avoid "asyncio.run() cannot be called from a running event loop"
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, _run())
        return future.result()
