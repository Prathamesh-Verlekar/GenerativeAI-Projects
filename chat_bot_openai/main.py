from fastapi import FastAPI, Request
from models.database import Base, engine
from routes import chat, history, index
from config.logging_config import logger

Base.metadata.create_all(bind=engine)
app = FastAPI(title="Advanced RAG Application with OpenAI gpt-4o-mini")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response Status: {response.status_code}")
    return response

app.include_router(chat.router)
app.include_router(history.router)
app.include_router(index.router)

@app.get("/")
def home():
    return {"message": "Welcome to Advanced RAG Application!"}
