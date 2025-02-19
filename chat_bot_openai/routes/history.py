from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from models.database import get_db
from models.chat_history import ChatHistory

router = APIRouter(prefix="/history", tags=["History"])

@router.get("/")
def get_history(db: Session = Depends(get_db)):
    return db.query(ChatHistory).all()
