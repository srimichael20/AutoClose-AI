"""Orchestrator - Coordinates Intake → Vision → Classification → MCP."""

import uuid
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from agents.intake_agent import IntakeAgent
from agents.vision_agent import VisionAgent
from agents.classification_agent import ClassificationAgent
from agents.mcp_agent import MCPAgent
from utils.schemas import (
    DocumentType,
    ProcessingStatus,
    WorkflowState,
    IntakeResult,
    VisionResult,
    ClassificationResult,
    MCPActionResult,
)


def _get_llm():
    from utils.config import get_settings

    s = get_settings()
    if s.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=s.openai_model, api_key=s.openai_api_key, temperature=0)
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=s.gemini_model, google_api_key=s.google_api_key, temperature=0)


class GraphState(TypedDict):
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


def _to_workflow(state: dict[str, Any]) -> WorkflowState:
    def opt(d, k, ctor):
        v = d.get(k)
        return ctor(**v) if v and isinstance(v, dict) else None

    return WorkflowState(
        job_id=state.get("job_id", ""),
        document_id=state.get("document_id", ""),
        file_path=state.get("file_path"),
        raw_content=state.get("raw_content"),
        document_type=DocumentType(state.get("document_type", "text")),
        intake_result=opt(state, "intake_result", IntakeResult),
        vision_result=opt(state, "vision_result", VisionResult),
        classification_result=opt(state, "classification_result", ClassificationResult),
        mcp_result=opt(state, "mcp_result", MCPActionResult),
        current_step=state.get("current_step", "intake"),
        status=ProcessingStatus(state.get("status", "pending")),
        error=state.get("error"),
        messages=state.get("messages", []),
        context_for_next_agent=state.get("context_for_next_agent", ""),
    )


def _to_updates(ws: WorkflowState) -> dict[str, Any]:
    u: dict[str, Any] = {
        "current_step": ws.current_step,
        "status": ws.status.value,
        "error": ws.error,
        "messages": ws.messages,
        "context_for_next_agent": ws.context_for_next_agent,
    }
    if ws.intake_result:
        u["intake_result"] = ws.intake_result.model_dump()
    if ws.vision_result:
        u["vision_result"] = ws.vision_result.model_dump()
    if ws.classification_result:
        u["classification_result"] = ws.classification_result.model_dump()
    if ws.mcp_result:
        u["mcp_result"] = ws.mcp_result.model_dump()
    return u


async def _intake(state: dict[str, Any]) -> dict[str, Any]:
    ws = _to_workflow(state)
    up = await IntakeAgent().process(ws)
    return _to_updates(WorkflowState(**{**ws.model_dump(), **up}))


async def _vision(state: dict[str, Any]) -> dict[str, Any]:
    ws = _to_workflow(state)
    up = await VisionAgent().process(ws)
    return _to_updates(WorkflowState(**{**ws.model_dump(), **up}))


async def _classification(state: dict[str, Any]) -> dict[str, Any]:
    ws = _to_workflow(state)
    up = await ClassificationAgent().process(ws)
    return _to_updates(WorkflowState(**{**ws.model_dump(), **up}))


async def _mcp(state: dict[str, Any]) -> dict[str, Any]:
    ws = _to_workflow(state)
    up = await MCPAgent().process(ws)
    return _to_updates(WorkflowState(**{**ws.model_dump(), **up}))


def _after_intake(state: dict[str, Any]) -> Literal["vision", "classification"]:
    if state.get("status") == "failed":
        return "classification"
    ir = state.get("intake_result")
    if ir and isinstance(ir, dict) and ir.get("requires_vision"):
        return "vision"
    return "classification"


def create_graph() -> StateGraph:
    g = StateGraph(GraphState)
    g.add_node("intake", _intake)
    g.add_node("vision", _vision)
    g.add_node("classification", _classification)
    g.add_node("mcp", _mcp)

    g.set_entry_point("intake")
    g.add_conditional_edges("intake", _after_intake)
    g.add_edge("vision", "classification")
    g.add_edge("classification", "mcp")
    g.add_edge("mcp", END)
    return g


_compiled = None


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = create_graph().compile()
    return _compiled


async def run_workflow(
    document_id: str,
    document_type: DocumentType,
    file_path: str | None = None,
    raw_content: str | None = None,
) -> dict[str, Any]:
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
    return await get_graph().ainvoke(initial, config={"configurable": {"thread_id": job_id}})
