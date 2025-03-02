from fastapi import APIRouter, Request
import os

router = APIRouter(prefix="/session", tags=["Session Management"])

@router.post("/create")
def create_session(request: Request):
    """
    Creates a new session if not already present.
    """
    if "session_id" not in request.session:
        request.session["session_id"] = os.urandom(24).hex()
    return {"message": "Session created successfully!", "session_id": request.session["session_id"]}

@router.get("/current")
def get_current_session(request: Request):
    """
    Retrieves the current session ID if it exists.
    """
    if "session_id" in request.session:
        return {"session_id": request.session["session_id"]}
    return {"message": "No active session found. Use /session/create to start one."}