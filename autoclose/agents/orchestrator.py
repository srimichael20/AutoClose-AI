"""Orchestrator Agent - LangGraph workflow coordination."""

import uuid
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from autoclose.agents.classification_agent import ClassificationAgent
from autoclose.agents.intake_agent import IntakeAgent
from autoclose.agents.mcp_integration_agent import MCPIntegrationAgent
from autoclose.agents.vision_agent import VisionAgent
from autoclose.config import get_settings
from autoclose.schemas import DocumentType, ProcessingStatus, WorkflowState


def get_llm():
    """Get configured LLM instance (OpenAI or Gemini)."""
    from langchain_core.language_models import BaseChatModel

    settings = get_settings()
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    )


class GraphState(TypedDict):
    """State schema for LangGraph - compatible with WorkflowState fields."""

    job_id: str
    document_id: str
    file_path: str | None
    raw_content: str | None
    document_type: str
    intake_result: dict | None
    vision_result: dict | None
    classification_result: dict | None
    mcp_result: dict | None
    current_step: str
    status: str
    error: str | None
    messages: list[str]
    context_for_next_agent: str


def _state_to_workflow(state: dict[str, Any]) -> WorkflowState:
    """Convert graph state dict to WorkflowState."""
    from autoclose.schemas import (
        ClassificationResult,
        IntakeResult,
        MCPActionResult,
        VisionResult,
    )

    intake = state.get("intake_result")
    if intake and isinstance(intake, dict):
        intake = IntakeResult(**intake)
    elif intake is None:
        intake = None

    vision = state.get("vision_result")
    if vision and isinstance(vision, dict):
        vision = VisionResult(**vision)
    elif vision is None:
        vision = None

    classification = state.get("classification_result")
    if classification and isinstance(classification, dict):
        classification = ClassificationResult(**classification)
    elif classification is None:
        classification = None

    mcp = state.get("mcp_result")
    if mcp and isinstance(mcp, dict):
        mcp = MCPActionResult(**mcp)
    elif mcp is None:
        mcp = None

    doc_type = state.get("document_type", "text")
    if isinstance(doc_type, str):
        doc_type = DocumentType(doc_type)

    return WorkflowState(
        job_id=state.get("job_id", ""),
        document_id=state.get("document_id", ""),
        file_path=state.get("file_path"),
        raw_content=state.get("raw_content"),
        document_type=doc_type,
        intake_result=intake,
        vision_result=vision,
        classification_result=classification,
        mcp_result=mcp,
        current_step=state.get("current_step", "intake"),
        status=ProcessingStatus(state.get("status", "pending")),
        error=state.get("error"),
        messages=state.get("messages", []),
        context_for_next_agent=state.get("context_for_next_agent", ""),
    )


def _workflow_to_state_updates(ws: WorkflowState) -> dict[str, Any]:
    """Convert WorkflowState to state update dict."""
    updates: dict[str, Any] = {
        "current_step": ws.current_step,
        "status": ws.status.value,
        "error": ws.error,
        "messages": ws.messages,
        "context_for_next_agent": ws.context_for_next_agent,
    }
    if ws.intake_result:
        updates["intake_result"] = ws.intake_result.model_dump()
    if ws.vision_result:
        updates["vision_result"] = ws.vision_result.model_dump()
    if ws.classification_result:
        updates["classification_result"] = ws.classification_result.model_dump()
    if ws.mcp_result:
        updates["mcp_result"] = ws.mcp_result.model_dump()
    return updates


async def _run_intake(state: dict[str, Any]) -> dict[str, Any]:
    """Intake node."""
    ws = _state_to_workflow(state)
    agent = IntakeAgent()
    updates = await agent.process(ws)
    return _workflow_to_state_updates(WorkflowState(**{**ws.model_dump(), **updates}))


async def _run_vision(state: dict[str, Any]) -> dict[str, Any]:
    """Vision node."""
    ws = _state_to_workflow(state)
    agent = VisionAgent()
    updates = await agent.process(ws)
    return _workflow_to_state_updates(WorkflowState(**{**ws.model_dump(), **updates}))


async def _run_classification(state: dict[str, Any]) -> dict[str, Any]:
    """Classification node."""
    ws = _state_to_workflow(state)
    agent = ClassificationAgent()
    updates = await agent.process(ws)
    return _workflow_to_state_updates(WorkflowState(**{**ws.model_dump(), **updates}))


async def _run_mcp(state: dict[str, Any]) -> dict[str, Any]:
    """MCP integration node."""
    ws = _state_to_workflow(state)
    agent = MCPIntegrationAgent()
    updates = await agent.process(ws)
    return _workflow_to_state_updates(WorkflowState(**{**ws.model_dump(), **updates}))


def _route_after_intake(
    state: dict[str, Any],
) -> Literal["vision", "classification"]:
    """Route: after intake, go to vision if requires_vision else classification."""
    if state.get("status") == "failed":
        return "classification"
    intake = state.get("intake_result")
    if intake and isinstance(intake, dict) and intake.get("requires_vision"):
        return "vision"
    return "classification"


def _route_after_vision(state: dict[str, Any]) -> Literal["classification"]:
    """Always go to classification after vision."""
    return "classification"


def _route_after_classification(state: dict[str, Any]) -> Literal["mcp_integration"]:
    """Always go to MCP after classification."""
    return "mcp_integration"


def create_workflow_graph() -> StateGraph:
    """
    Create the LangGraph workflow:
    intake -> (vision? ->) classification -> mcp_integration -> END
    """
    graph = StateGraph(GraphState)

    graph.add_node("intake", _run_intake)
    graph.add_node("vision", _run_vision)
    graph.add_node("classification", _run_classification)
    graph.add_node("mcp_integration", _run_mcp)

    graph.set_entry_point("intake")
    graph.add_conditional_edges("intake", _route_after_intake)
    graph.add_edge("vision", "classification")
    graph.add_edge("classification", "mcp_integration")
    graph.add_edge("mcp_integration", END)

    return graph


def get_compiled_graph():
    """Return compiled LangGraph workflow."""
    return create_workflow_graph().compile()


async def run_workflow(
    document_id: str,
    document_type: DocumentType,
    file_path: str | None = None,
    raw_content: str | None = None,
) -> dict[str, Any]:
    """
    Execute the full workflow for a single document.
    Returns final state as dict.
    """
    job_id = str(uuid.uuid4())
    initial: GraphState = {
        "job_id": job_id,
        "document_id": document_id,
        "file_path": file_path,
        "raw_content": raw_content,
        "document_type": document_type.value,
        "intake_result": None,
        "vision_result": None,
        "classification_result": None,
        "mcp_result": None,
        "current_step": "intake",
        "status": "pending",
        "error": None,
        "messages": [],
        "context_for_next_agent": "",
    }

    app = get_compiled_graph()
    config = {"configurable": {"thread_id": job_id}}
    final = await app.ainvoke(initial, config=config)
    return final
