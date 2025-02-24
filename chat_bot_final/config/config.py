import os
import secrets

OPENAI_MODEL_NAME = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
PINECONE_ENVIRONMENT = "us-east-1"
PINECONE_CHAT_INDEX = "chat-history"
PINECONE_VECTOR_INDEX = "vector-store"
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", secrets.token_hex(32))
SESSION_EXPIRY = 3600 
UPLOAD_DIR = "uploaded_files"
