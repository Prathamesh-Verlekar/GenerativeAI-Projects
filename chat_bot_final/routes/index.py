<<<<<<< HEAD
from fastapi import APIRouter, UploadFile, File, Request, Query
=======
from fastapi import APIRouter, UploadFile, File, Request, HTTPException
>>>>>>> e22de01 (Session backend)
from schemas.chat_schemas import FileUploadResponse
from services.indexer import process_uploaded_files

router = APIRouter(prefix="/index", tags=["Indexing"])

@router.post("/", response_model=FileUploadResponse)
<<<<<<< HEAD
async def index_uploaded_files(request: Request, files: list[UploadFile] = File(...), session_id: str = Query(None)):
    """
    Handles file uploads and indexes them per session.
    """
    session_id = session_id or request.session.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Session not found. Use /session/create to start one.")
=======
async def index_uploaded_files(request: Request, files: list[UploadFile] = File(...)):
    """
    Handles file uploads and indexes them using session_id.
    """
    session_id = request.session.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Session not found. Create a session using /session/create.")
>>>>>>> e22de01 (Session backend)
    response = process_uploaded_files(files, session_id)
    return FileUploadResponse(session_id=session_id, message=response["message"])
