# from fastapi import APIRouter, Depends
# from sqlalchemy.orm import Session
# from services.llm import generate_answer
# from models.database import get_db
# from models.chat_history import ChatHistory

# router = APIRouter(prefix="/chat", tags=["Chat"])

# @router.post("/")
# def ask_question(query: str, db: Session = Depends(get_db)):
#     response = generate_answer(query)
    
#     # Save to DB
#     new_chat = ChatHistory(question=query, answer=response)
#     db.add(new_chat)
#     db.commit()

#     return {"question": query, "answer": response}

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from services.llm import generate_answer
from models.database import get_db

# Initialize Router
router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/")
def ask_question(query: str, db: Session = Depends(get_db)):
    """
    Processes user queries by retrieving the last 4 interactions,
    generating an answer using the LLM, and storing it in the database.
    """
    response = generate_answer(query, db)
    return response

