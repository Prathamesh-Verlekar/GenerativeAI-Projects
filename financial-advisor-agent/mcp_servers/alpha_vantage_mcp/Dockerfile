FROM python:3.11-slim

WORKDIR /srv
COPY mcp_servers/alpha_vantage_mcp /srv

# Official MCP install uses extras like mcp[cli] (safe to use even if you don't run CLI)
RUN pip install --no-cache-dir "mcp[cli]" requests redis starlette uvicorn

EXPOSE 8787
CMD ["python", "server.py"]

