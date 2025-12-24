<<<<<<< HEAD
from fastapi import APIRouter, Request, Query
from fastapi import HTTPException
=======
from fastapi import APIRouter, Request, HTTPException
>>>>>>> e22de01 (Session backend)
from schemas.chat_schemas import ChatRequest, ChatResponse
from services.llm import generate_answer

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/", response_model=ChatResponse)
<<<<<<< HEAD
async def chat_with_bot(request: Request, chat_request: ChatRequest, session_id: str = Query(None)):
    """Chat endpoint that handles user query and manages session context."""
    session_id = session_id or request.session.get("session_id")
=======
async def chat_with_bot(request: Request, chat_request: ChatRequest):
    """
    Handles user query using session_id from the session.
    """
    session_id = request.session.get("session_id")
>>>>>>> e22de01 (Session backend)
    if not session_id:
        raise HTTPException(status_code=400, detail="Session not found. Create a session using /session/create.")

    response = generate_answer(chat_request.query, session_id)
    return ChatResponse(session_id=session_id, question=chat_request.query, answer=response["answer"])
