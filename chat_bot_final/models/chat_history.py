from sqlalchemy import Column, Integer, String
from models.database import Base

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, nullable=False)
    question = Column(String, nullable=False)
    answer = Column(String, nullable=False)
