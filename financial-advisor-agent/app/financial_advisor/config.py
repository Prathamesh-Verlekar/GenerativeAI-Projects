from __future__ import annotations

from pydantic import BaseModel
import os


class Settings(BaseModel):
    # LLM
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "openai/gpt-4.1-mini")

    # MCP endpoints (SSE)
    MCP_ALPHA_URL: str = os.getenv("MCP_ALPHA_URL", "http://localhost:8787/sse")
    MCP_SHEETS_URL: str = os.getenv("MCP_SHEETS_URL", "http://localhost:8790/sse")

    # Optional: if your Sheets MCP server reads default spreadsheet ID from env,
    # your app doesn’t need this. Keep it here if you want app-side routing.
    GOOGLE_SHEETS_DEFAULT_RANGE: str = os.getenv("GOOGLE_SHEETS_DEFAULT_RANGE","Sheet1!A1:Z200")

    # Context limits
    SHEETS_CONTEXT_MAX_ROWS: int = int(os.getenv("SHEETS_CONTEXT_MAX_ROWS", "60"))
    SHEETS_CONTEXT_MAX_CHARS: int = int(os.getenv("SHEETS_CONTEXT_MAX_CHARS", "12000"))


settings = Settings()

