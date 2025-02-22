from fastapi import APIRouter, Query
from schemas.chat_schemas import HistoryResponse, ChatResponse
from services.llm import retrieve_history

router = APIRouter(prefix="/history", tags=["History"])

@router.get("/", response_model=HistoryResponse)
async def get_chat_history(session_id: str = Query(..., description="Session ID for which to retrieve chat history")):
    """Retrieve chat history for a specific session ID."""
    history_records = retrieve_history(session_id)
    history = [ChatResponse(session_id=session_id, **record) for record in history_records]

    return HistoryResponse(session_id=session_id, history=history)


