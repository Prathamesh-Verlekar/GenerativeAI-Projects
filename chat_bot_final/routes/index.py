from fastapi import APIRouter, UploadFile, File, Request
from schemas.chat_schemas import FileUploadResponse
from services.indexer import process_uploaded_files
import os

router = APIRouter(prefix="/index", tags=["Indexing"])

@router.post("/", response_model=FileUploadResponse)
async def index_uploaded_files(request: Request, files: list[UploadFile] = File(...)):
    """Upload files and index them with session metadata."""
    session_id = request.session.get("session_id")
    if not session_id:
        session_id = os.urandom(24).hex()
        request.session["session_id"] = session_id

    response = process_uploaded_files(files, session_id)
    return FileUploadResponse(session_id=session_id, message=response["message"])


