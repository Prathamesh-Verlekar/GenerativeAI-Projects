from fastapi import FastAPI, Request
from models.database import Base, engine
from routes import chat, history, index
import logging
from config.logging_config import logger

# Initialize Database
Base.metadata.create_all(bind=engine)

# Initialize FastAPI
app = FastAPI(title="Optimized Chatbot with Logging")

# Middleware to Log All API Requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response Status: {response.status_code}")
    return response

# Include Routes
app.include_router(chat.router)
app.include_router(history.router)
app.include_router(index.router)

@app.get("/")
def home():
    logger.info("Home endpoint accessed")
    return {"message": "Hello!"}

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected Error: {str(exc)}", exc_info=True)
    return {"error": "Internal Server Error"}
