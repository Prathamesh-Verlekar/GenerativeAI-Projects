# 📈 Agentic Financial Advisor
**ADK · MCP · Alpha Vantage · Redis · OpenAI · Streamlit**

An **agentic financial advisor system** built using **Google ADK**, **Model Context Protocol (MCP)**, **Alpha Vantage market data**, **Redis caching**, **OpenAI GPT models**, and a **Streamlit chat UI**.

This project demonstrates **production-grade agent architecture** with:
- Tool-augmented LLM reasoning  
- Live financial market data  
- Guardrails & compliance-safe responses  
- Observability (tool calls, latency, cache hits)  
- Session memory & conversational follow-ups  
- Visual chat interface

> ⚠️ **Disclaimer**  
> This project is for **educational and demonstration purposes only** and does **not** constitute financial advice.

---

## ✨ Key Features

### 🧠 Agentic Reasoning
- LLM-powered financial assistant using **OpenAI GPT (via LiteLLM)**
- Uses tools **only when required**
- Guardrails prevent prompt injection & unsafe financial claims

### 🔌 MCP-Based Tooling
- Market data and spreadsheets exposed via **Model Context Protocol (MCP)**
- Tools available:
  - `search_ticker`
  - `get_quote`
  - `get_history`
  - `fetch_values` (Google Sheets)
  - `list_rows` (Google Sheets)
  - `append_row` (Google Sheets)
  - `update_range` (Google Sheets)

### ⚡ Performance & Caching
- **Redis-backed caching** for Alpha Vantage API calls
- Cache hit/miss tracking
- TTL-based invalidation

### 📊 Observability
- Tool-level logging:
  - tool name
  - symbol / query
  - cache hit or miss
  - latency (ms)
- Logs accessible via Docker

### 🧠 Session Memory
- Remembers last referenced ticker
- Supports follow-ups like:
  - “What is its price?”
  - “What about yesterday?”

### 💬 Visual Chat UI
- Streamlit-based chat interface
- Sidebar shows:
  - session ID
  - last remembered ticker
- Real-time interaction with the agent

---

## 📂 Repository Structure

```text
financial-advisor-agent/
│
├── app/
│   └── financial_advisor/
│       ├── agent.py                  # ADK agent definition
│       ├── config.py                 # Environment-based config
│       ├── prompts/
│       │   └── system.md             # System prompt (instructions)
│       ├── guardrails/
│       │   ├── input_validation.py   # Prompt injection protection
│       │   ├── output_validation.py  # Safe financial phrasing
│       │   └── tool_policy.py        # Tool usage policy
│       └── tools/
│           └── mcp_alpha_vantage.py  # MCP toolset wiring
│
├── mcp_servers/
│   ├── alpha_vantage_mcp/
│   │   ├── server.py                 # MCP Server (FastMCP + SSE)
│   │   ├── alpha_client.py           # Alpha Vantage API client
│   │   ├── cache.py                  # Redis cache wrapper
│   │   └── Dockerfile
│   └── google_sheets_mcp/
│       ├── server.py                 # Google Sheets MCP Server
│       ├── sheets_client.py          # Service Account client wrapper
│       └── Dockerfile
│
├── streamlit_app.py                  # Visual chat UI
├── run_agent_cli.py                  # CLI chat runner
├── docker-compose.yml                # Redis + MCP services
├── .env.example
└── README.md
```

---

## 🔄 End-to-End Execution Flow

```mermaid
flowchart TD
    A[User Input] --> B[Streamlit / CLI]
    B --> C[Guardrails: Input Validation]
    C --> D[ADK Runner]
    D --> E[LLM Reasoning]

    E -->|Needs Data| F[MCP Toolset]
    F -->|SSE| G[MCP Server]

    G --> H{Cache Hit?}
    H -->|Yes| I[Redis]
    H -->|No| J[Alpha Vantage API]

    I --> K[Tool Result]
    J --> K[Tool Result]

    K --> L[LLM Synthesis]
    L --> M[Guardrails: Output Validation]
    M --> N[User Response]
```

---

## 🧠 Agent Design (Google ADK)

- Uses **`LlmAgent`**
- Model: **OpenAI GPT via LiteLLM**
- Instruction loaded from `system.md`
- Tools connected via **`McpToolset`**

### Guardrails Applied

| Stage | Guardrail |
|----|----|
| Input | Normalize text, detect prompt injection |
| Tool Use | Restricted MCP toolset |
| Output | No guarantees, compliant financial language |

---

## 🔌 MCP Server Design

### Stack
- FastMCP
- SSE (Server-Sent Events)
- Starlette + Uvicorn
- Redis

### Tools
- `search_ticker(query)`
- `get_quote(symbol)`
- `get_history(symbol, days)`

### Observability (Server Logs)

```text
[MCP] get_quote symbol=AAPL cache=HIT latency_ms=3.1
[MCP] get_history symbol=AAPL days=5 cache=MISS latency_ms=412.7
```

### Google Sheets MCP Setup

- Uses a **service account** for authentication.
- Required environment variable: `GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON` (the full JSON for the service account).
- Optional default: `GOOGLE_SHEETS_SPREADSHEET_ID` to avoid repeating the spreadsheet ID in tool calls.
- Tools exposed:
  - `fetch_values(range_a1, spreadsheet_id?, worksheet_title?)`
  - `list_rows(spreadsheet_id?, worksheet_title?, limit=20)`
  - `append_row(row, spreadsheet_id?, worksheet_title?)`
  - `update_range(range_a1, values, spreadsheet_id?, worksheet_title?)`
- Default port: **8790** (see `docker-compose.yml`).

### Google Sheets MCP: End-to-End Setup

1. **Create a Google Cloud service account** with the **Google Sheets API** enabled.
2. **Generate a service account key** (JSON).
3. **Share your target spreadsheet** with the service account email (usually `<service-account-name>@<project>.iam.gserviceaccount.com`).
4. **Set environment variables** (example for `.env` or docker-compose overrides):
   ```env
   # Required: raw JSON from the downloaded key file
   GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON='{"type":"service_account","project_id":"...","private_key_id":"...","private_key":"-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n","client_email":"<service-account>@<project>.iam.gserviceaccount.com","client_id":"...","token_uri":"https://oauth2.googleapis.com/token"}'

   # Optional: default spreadsheet used when an ID is not passed to tools
   GOOGLE_SHEETS_SPREADSHEET_ID=1abc2DefGhijklmNoPqRstuVWxyz0123456789
   ```
   - If you prefer to keep the key in a file, load it when exporting: `export GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON="$(cat /path/key.json | tr -d '\n')"`.
5. **Start the server**
   - Docker Compose (recommended): `docker compose up google-sheets-mcp`
   - Local dev: `uvicorn mcp_servers.google_sheets_mcp.server:app --port 8790 --host 0.0.0.0`
6. **Point the agent at the MCP endpoint**
   - Set `MCP_SHEETS_URL=http://localhost:8790/sse` (or the container hostname when networked via Docker).
   - With `GOOGLE_SHEETS_SPREADSHEET_ID` set, tool calls only need ranges and values.

---

## 🧠 Session Memory

- Maintained per session (CLI or Streamlit)
- Extracted from agent responses (e.g. “ticker: AAPL”)
- Enables natural follow-ups

### Example
```
User: Find ticker for CoreWeave
Agent: The ticker code is CRWV

User: What is its price?
→ Automatically resolved as CRWV

User: What about yesterday?
→ Uses CRWV history
```

---

## ⚡ Redis Caching Strategy

| Data Type | TTL |
|---------|----|
| Quotes | ~20 seconds |
| History | 60+ seconds |
| Search | 60 seconds |

Redis keys:
```text
quote:AAPL
hist:AAPL:5
search:tesla
```

---

## 💬 Streamlit Chat UI

### Features
- Chat-style UI
- Sidebar session info
- Reset session
- Live interaction with agent

Run:
```bash
streamlit run streamlit_app.py
```

---

## 🐳 Running the System

### 1️⃣ Start MCP + Redis
```bash
docker compose up
```

### 2️⃣ Run Streamlit App
```bash
source .venv/bin/activate
set -a; source .env; set +a
streamlit run streamlit_app.py
```

### 3️⃣ View MCP Logs
```bash
docker logs -f financial-advisor-agent-alpha-mcp-1
```

---

## 🔐 Environment Variables

```env
ALPHAVANTAGE_API_KEY=xxxx
OPENAI_API_KEY=sk-xxxx
OPENAI_MODEL=openai/gpt-4.1-mini

MCP_ALPHA_URL=http://localhost:8787/sse
MCP_SHEETS_URL=http://localhost:8790/sse
REDIS_URL=redis://redis:6379/0
QUOTE_CACHE_TTL_SEC=20
MAX_HISTORY_DAYS=3650

# Google Sheets MCP
GOOGLE_SHEETS_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'
GOOGLE_SHEETS_SPREADSHEET_ID=1abc2DefGhijklmNoPqRstuVWxyz0123456789
```

---

## 🚀 Future Enhancements

- Persistent session memory (Redis / DB)
- Cost tracking (LLM + API usage)
- Prometheus / OpenTelemetry metrics
- Portfolio analysis agent
- Multi-agent orchestration (Planner / Analyst / Compliance)
- Cloud Run / GKE deployment

---

## 🧩 Why This Architecture Matters

This project demonstrates:
- **Agentic AI design patterns**
- **Tool-first LLM reasoning**
- **Protocol-based integration (MCP)**
- **Enterprise-grade observability**
- **Compliance-aware guardrails**

It closely mirrors **real-world AI platform architectures** used in:
- Finance
- Consulting
- Internal AI copilots
- Research & analytics systems

