from pydantic import BaseModel, Field
from typing import Optional, List


class ChatRequest(BaseModel):
    """Request body model for chat endpoint"""
    query: str = Field(..., example="What is Retrieval-Augmented Generation (RAG)?")
    temperature: Optional[float] = Field(default=0.7, example=0.7, description="Temperature for LLM response")
    top_k: Optional[int] = Field(default=5, example=5, description="Top K documents to retrieve")


class ChatResponse(BaseModel):
    """Response body model for chat endpoint"""
    session_id: str
    question: str
    answer: str


class FileUploadResponse(BaseModel):
    """Response model after file upload and indexing"""
    session_id: str
    message: str


class HistoryResponse(BaseModel):
    """Response model for chat history retrieval"""
    session_id: str
    history: List[ChatResponse]
