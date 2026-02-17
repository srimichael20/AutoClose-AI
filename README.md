# AutoClose AI

**Autonomous Accounting Agent for SMB Book Closing** – Streamlit-based multi-agent dashboard with real-time workflow execution.

## Architecture

```
Intake → Vision → Classification → MCP → Summary
```

| Agent | Role |
|-------|------|
| **Intake** | Multi-modal document intake (text, PDF, images) |
| **Vision** | OCR + document extraction |
| **Classification** | LLM + vector similarity for transaction classification |
| **MCP** | Database, file storage, API, notifications |
| **Summary** | AI-generated financial summary |

## Project Structure

```
autoclose-ai/
├── agents/
│   ├── intake_agent.py
│   ├── vision_agent.py
│   ├── classification_agent.py
│   ├── mcp_agent.py
│   ├── summary_agent.py
│   ├── workflow_runner.py
│   └── orchestrator.py
├── vector_db/
│   └── chroma_store.py
├── database/
│   └── sqlite_db.py
├── api/
│   └── routes.py
├── utils/
│   ├── config.py
│   ├── schemas.py
│   ├── embeddings.py
│   ├── file_storage.py
│   ├── notification.py
│   └── api_client.py
├── streamlit_app.py    # Main UI
├── app.py              # FastAPI (optional)
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # Set OPENAI_API_KEY or GOOGLE_API_KEY
streamlit run streamlit_app.py
```

Open http://localhost:8501

## Streamlit UI Features

- **Drag-and-drop file upload** – PDF, images, text
- **Text input** – Paste document content or prompts
- **User prompt** – e.g. "Close monthly books"
- **Run Workflow** – Execute full agent pipeline
- **Agent logs** – Step-by-step workflow status
- **Extracted content** – Document preview
- **Classification results** – Category, amount, description
- **MCP actions** – DB, file, API, notifications
- **Financial summary** – AI-generated summary

## Environment

| Variable | Description |
|----------|-------------|
| `LLM_PROVIDER` | `openai` or `gemini` |
| `OPENAI_API_KEY` | OpenAI API key |
| `GOOGLE_API_KEY` | Google API key (Gemini) |
| `CHROMA_PERSIST_DIRECTORY` | Vector DB path |
| `DATABASE_PATH` | SQLite path |

## License

MIT
