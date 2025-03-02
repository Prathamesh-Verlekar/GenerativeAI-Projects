from fastapi import APIRouter, UploadFile, File, Request, Query
from schemas.chat_schemas import FileUploadResponse
from services.indexer import process_uploaded_files
import os

router = APIRouter(prefix="/index", tags=["Indexing"])

@router.post("/", response_model=FileUploadResponse)
async def index_uploaded_files(request: Request, files: list[UploadFile] = File(...), session_id: str = Query(None)):
    """
    Handles file uploads and indexes them per session.
    """
    session_id = session_id or request.session.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Session not found. Use /session/create to start one.")
    response = process_uploaded_files(files, session_id)
    return FileUploadResponse(session_id=session_id, message=response["message"])


