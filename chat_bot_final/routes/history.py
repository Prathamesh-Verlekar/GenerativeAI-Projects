from fastapi import APIRouter, Query, Request
from schemas.chat_schemas import HistoryResponse, ChatResponse
from services.llm import retrieve_history
import logging

router = APIRouter(prefix="/history", tags=["History"])

@router.get("/", response_model=HistoryResponse)
async def get_chat_history(request: Request, session_id: str = Query(None)):
    """
    Retrieves chat history for the active session.
    """
    session_id = session_id or request.session.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Session not found. Use /session/create to start one.")
    
    logging.info(f"Fetching history for session_id: {session_id}")
    
    history_records = retrieve_history(session_id)
    
    if not history_records:
        logging.warning(f"No history found for session_id: {session_id}")
        return HistoryResponse(session_id=session_id, history=[])
    
    history = [ChatResponse(session_id=session_id, **record) for record in history_records]
    
    logging.info(f"Retrieved {len(history)} records for session_id: {session_id}")
    
    return HistoryResponse(session_id=session_id, history=history)


