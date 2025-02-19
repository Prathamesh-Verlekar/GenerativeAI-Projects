from fastapi import APIRouter
from services.llm import generate_answer

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/")
def chat_endpoint(question: str):
    """API endpoint to generate response from the model without nested JSON."""
    response = generate_answer(question)
    return response

