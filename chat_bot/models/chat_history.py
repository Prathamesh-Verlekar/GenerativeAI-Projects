# from sqlalchemy import Column, Integer, Text
# from models.database import Base

# class ChatHistory(Base):
#     __tablename__ = "chat_history"
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     question = Column(Text, nullable=False)
#     answer = Column(Text, nullable=False)

from sqlalchemy import Column, Integer, String
from models.database import Base

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, nullable=False)
    answer = Column(String, nullable=False)
