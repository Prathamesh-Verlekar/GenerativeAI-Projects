from fastapi import APIRouter, Request
from fastapi import HTTPException
from schemas.chat_schemas import ChatRequest, ChatResponse
from services.llm import generate_answer
import os

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/", response_model=ChatResponse)
async def chat_with_bot(request: Request, chat_request: ChatRequest):
    """Chat endpoint that handles user query and manages session context."""
    session_id = request.session.get("session_id")
    if not session_id:
        session_id = os.urandom(24).hex()
        request.session["session_id"] = session_id

    response = generate_answer(chat_request.query, session_id)

    # Handle missing 'answer' key
    if "answer" not in response:
        raise HTTPException(status_code=500, detail=f"Failed to generate response. Details: {response.get('error', 'Unknown error.')}")

    return ChatResponse(
        session_id=session_id,
        question=chat_request.query,
        answer=response["answer"]
    )
