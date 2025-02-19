from fastapi import APIRouter, UploadFile, File
from services.indexer import process_uploaded_files

router = APIRouter(prefix="/index", tags=["Indexing"])

@router.post("/")
def index_uploaded_files(files: list[UploadFile] = File(...)):
    response = process_uploaded_files(files)
    return response
