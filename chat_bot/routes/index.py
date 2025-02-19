from fastapi import APIRouter, UploadFile, File
from services.indexer import process_uploaded_files
from services.vectorstore import vector_db

router = APIRouter(prefix="/index", tags=["Indexing"])

@router.post("/")
def index_uploaded_files(files: list[UploadFile] = File(...)):
    """API to upload and index multiple files."""
    response = process_uploaded_files(files)
    return response

@router.get("/view")
def view_indexed_documents():
    """Retrieve all indexed documents from ChromaDB."""
    collection = vector_db._collection 
    stored_docs = collection.get()

    return {
        "ids": stored_docs["ids"],
        "documents": stored_docs["documents"],
        "metadata": stored_docs["metadatas"]
    }
