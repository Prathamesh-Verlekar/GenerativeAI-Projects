# from fastapi import APIRouter, Depends
# from sqlalchemy.orm import Session
# from models.database import get_db
# from models.chat_history import ChatHistory

# router = APIRouter(prefix="/history", tags=["History"])

# @router.get("/")
# def get_chat_history(db: Session = Depends(get_db)):
#     chats = db.query(ChatHistory).all()
#     return [{"id": chat.id, "question": chat.question, "answer": chat.answer} for chat in chats]

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from models.database import get_db
from models.chat_history import ChatHistory

# Initialize Router
router = APIRouter(prefix="/history", tags=["History"])

@router.get("/")
def get_chat_history(db: Session = Depends(get_db)):
    """
    Fetches the last 10 chat interactions from the database.
    """
    chats = db.query(ChatHistory).order_by(ChatHistory.id.desc()).limit(10).all()
    return [{"question": chat.question, "answer": chat.answer} for chat in reversed(chats)]

