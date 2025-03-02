from fastapi import FastAPI, Request
from starlette.middleware.sessions import SessionMiddleware
from routes import chat, index, history, session
from config.config import SESSION_SECRET_KEY
from config.logging_config import logger
import os

app = FastAPI(title="Session-Based Chatbot with Pinecone")

# Add SessionMiddleware (Use a secure secret key in production)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response Status: {response.status_code}")
    return response

# Include Routes
app.include_router(chat.router)
app.include_router(index.router)
app.include_router(history.router)
app.include_router(session.router)

@app.get("/")
def home(request: Request):
    if "session_id" not in request.session:
        request.session["session_id"] = os.urandom(24).hex()  # Generate unique session ID
    return {"message": "Hello!", "session_id": request.session["session_id"]}


