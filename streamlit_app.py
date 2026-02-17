"""
AutoClose AI - Streamlit Dashboard
Professional enterprise UI for Autonomous Accounting Agent.
"""

import uuid
from pathlib import Path

import streamlit as st

from agents.workflow_runner import run_workflow_sync
from utils.file_storage import FileStorage
from utils.schemas import DocumentType

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
WEBHOOK_URL = st.secrets["NOTIFICATION_WEBHOOK_URL"]

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="AutoClose AI | Accounting Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional CSS with animations and modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Hero section */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        font-size: 2.25rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
        letter-spacing: -0.025em;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    /* Input cards */
    .input-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    .input-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
    }
    
    /* Run button enhancement */
    .stButton > button {
        width: 100%;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 10px;
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
        border: none !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.4) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5) !important;
    }
    
    /* Agent log items with animation */
    .agent-log-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        margin: 6px 0;
        border-radius: 10px;
        background: #f8fafc;
        border-left: 4px solid #94a3b8;
        font-size: 0.9rem;
        animation: slideIn 0.3s ease-out;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-10px); }
        to { opacity: 1; transform: translateX(0); }
    }
    .agent-log-item.done {
        border-left-color: #22c55e;
        background: black;
    }
    .agent-log-item.running {
        border-left-color: #f59e0b;
        background: black;
    }
    .agent-log-item .step-icon {
        font-size: 1.25rem;
    }
    
    /* Result cards */
    .result-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .result-card h4 {
        color: #1e293b;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    
    /* Summary highlight box */
    .summary-box {
        padding: 1.5rem 2rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        color: #f8fafc;
        font-size: 1.05rem;
        line-height: 1.7;
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* Metric cards */
    .metric-card {
        background: black;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        border: 1px solid #e2e8f0;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


def _doc_type_from_filename(name: str) -> DocumentType:
    ext = (Path(name).suffix or "").lower()
    if ext == ".pdf":
        return DocumentType.PDF
    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}:
        return DocumentType.IMAGE
    return DocumentType.TEXT


def init_session():
    if "workflow_logs" not in st.session_state:
        st.session_state.workflow_logs = []
    if "workflow_result" not in st.session_state:
        st.session_state.workflow_result = None


def main():
    init_session()

    # Hero section
    st.markdown("""
    <div class="hero">
        <h1 class="hero-title">AutoClose AI</h1>
        <p class="hero-subtitle">Autonomous Accounting Agent · Multi-modal document processing</p>
        <span class="hero-badge">Intake → Vision → Classification → MCP → Summary</span>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        user_prompt = st.text_area(
            "User prompt",
            placeholder="e.g., Close monthly books, Process Q3 expenses",
            height=100,
            help="Optional context for the AI (e.g., 'Close monthly books')",
        )
        st.divider()
        st.markdown("**Workflow steps**")
        steps = ["1. Intake", "2. Vision (OCR)", "3. Classification", "4. MCP", "5. Summary"]
        for s in steps:
            st.markdown(f"• {s}")
        st.divider()
        st.caption("Powered by LangChain · Chroma · LLM")

    # Main content
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### 📁 Input")
        input_tab1, input_tab2 = st.tabs(["📤 File upload", "✏️ Text input"])

        file_path = None
        raw_content = None
        doc_type = DocumentType.TEXT

        with input_tab1:
            uploaded = st.file_uploader(
                "Drag and drop PDF, image, or text document",
                type=["pdf", "png", "jpg", "jpeg", "txt", "gif", "bmp", "tiff"],
                help="Supports PDF, images (PNG, JPG, etc.), and plain text",
            )
            if uploaded:
                fs = FileStorage()
                doc_id = str(uuid.uuid4())
                content = uploaded.read()
                dt = _doc_type_from_filename(uploaded.name)
                stored = fs.store_upload_sync(doc_id, content, dt)
                file_path = stored
                doc_type = dt
                st.success(f"✅ Uploaded: **{uploaded.name}** ({len(content):,} bytes)")

        with input_tab2:
            text_input = st.text_area(
                "Paste document content or transaction text",
                placeholder="e.g., Invoice from Staples: Office supplies $150.00. Date: 2024-01-15...",
                height=150,
            )
            if text_input:
                raw_content = text_input
                doc_type = DocumentType.TEXT

    with col2:
        st.markdown("### ▶️ Run")
        run_clicked = st.button("Run Workflow", type="primary", use_container_width=True)
        st.caption("Execute the full multi-agent pipeline")

    # Run workflow
    if run_clicked:
        if not file_path and not raw_content:
            st.error("⚠️ Provide a file or text input first.")
            st.stop()

        doc_id = str(uuid.uuid4())
        workflow_logs = []

        def on_step(step: str, status: str, data: dict):
            workflow_logs.append({"step": step, "status": status, "data": data})

        with st.spinner("🔄 Running multi-agent workflow..."):
            try:
                result = run_workflow_sync(
                    document_id=doc_id,
                    document_type=doc_type,
                    file_path=file_path,
                    raw_content=raw_content,
                    user_prompt=user_prompt or None,
                    on_step=on_step,
                )
            except Exception as e:
                st.error(f"Workflow error: {str(e)}")
                st.exception(e)
                st.stop()

        st.session_state.workflow_logs = workflow_logs
        st.session_state.workflow_result = result
        st.rerun()

    # Display results
    if st.session_state.workflow_result:
        result = st.session_state.workflow_result
        st.markdown("---")
        st.markdown("## 📋 Results")

        # Header with New run button
        rcol1, rcol2 = st.columns([4, 1])
        with rcol2:
            if st.button("🔄 New run", key="new_run", use_container_width=True):
                st.session_state.workflow_result = None
                st.session_state.workflow_logs = []
                st.rerun()

        # Agent workflow logs
        with st.expander("📜 Agent workflow logs", expanded=True):
            for entry in st.session_state.workflow_logs:
                css = "done" if entry["status"] == "done" else "running"
                icon = "✅" if entry["status"] == "done" else "⏳"
                st.markdown(
                    f'<div class="agent-log-item {css}">'
                    f'<span class="step-icon">{icon}</span>'
                    f'<span><strong>{entry["step"].upper()}</strong> – {entry["status"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Metrics row
        cr = result.get("classification_result")
        if cr:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Category", cr.get("category", "N/A"))
            with m2:
                amt = cr.get("amount")
                st.metric("Amount", f"${amt:,.2f}" if amt is not None else "N/A")
            with m3:
                st.metric("Confidence", f"{cr.get('confidence', 0):.0%}")

        # Tabs: Extracted | Classification | MCP | Summary
        r1, r2, r3, r4 = st.tabs([
            "📄 Extracted content",
            "🏷️ Classification",
            "🔌 MCP actions",
            "📊 Financial summary",
        ])

        with r1:
            ir = result.get("intake_result")
            vr = result.get("vision_result")
            content_preview = ""
            if vr:
                content_preview = vr.get("extracted_text", "")[:3000]
                if vr.get("structured_data"):
                    st.json(vr["structured_data"])
            elif ir and ir.get("raw_content"):
                content_preview = ir["raw_content"][:3000]
            if content_preview:
                st.text_area("Document preview", content_preview, height=220, disabled=True, key="preview")
            else:
                st.info("No extracted content to display.")

        with r2:
            cr = result.get("classification_result")
            if cr:
                st.markdown("**Category:** `{}`".format(cr.get("category", "N/A")))
                st.markdown("**Subcategory:** {}".format(cr.get("subcategory") or "N/A"))
                st.markdown("**Amount:** {}".format(cr.get("amount") if cr.get("amount") is not None else "N/A"))
                st.markdown("**Description:** {}".format(cr.get("description") or "N/A"))
                st.markdown("**Confidence:** {:.2f}".format(cr.get("confidence", 0)))
                if cr.get("reasoning"):
                    st.caption("Reasoning: {}".format(cr["reasoning"]))
            else:
                st.info("No classification result.")

        with r3:
            mcp = result.get("mcp_result")
            if mcp:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Database", "✅ Recorded" if mcp.get("database_recorded") else "❌")
                with c2:
                    st.metric("File stored", "✅" if mcp.get("file_stored") else "❌")
                with c3:
                    st.metric("API called", "✅" if mcp.get("api_called") else "❌")
                with c4:
                    st.metric("Notification", "✅ Sent" if mcp.get("notification_sent") else "⏭️ Skip")
                if mcp.get("details"):
                    with st.expander("Details"):
                        st.json(mcp["details"])
            else:
                st.info("No MCP actions.")

        with r4:
            sr = result.get("summary_result")
            if sr:
                summary_text = sr.get("financial_summary", "No summary generated.")
                st.markdown(
                    f'<div class="summary-box">{summary_text}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("No summary available.")

        if result.get("error"):
            st.error("Error: {}".format(result["error"]))


if __name__ == "__main__":
    main()
