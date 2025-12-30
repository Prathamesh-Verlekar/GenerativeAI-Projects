FROM python:3.11-slim

WORKDIR /srv
COPY mcp_servers/google_sheets_mcp /srv

RUN pip install --no-cache-dir "mcp[cli]" gspread google-auth starlette uvicorn

CMD ["python", "server.py"]
