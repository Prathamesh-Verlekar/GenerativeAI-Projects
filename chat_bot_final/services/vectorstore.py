import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores.pinecone import Pinecone as LangchainPinecone
from langchain_openai import OpenAIEmbeddings
from config.config import PINECONE_API_KEY, PINECONE_CHAT_INDEX, PINECONE_VECTOR_INDEX, OPENAI_EMBEDDING_MODEL

PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Index for chat history
if PINECONE_CHAT_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_CHAT_INDEX,
        dimension=1536,  # Assuming OpenAI embedding dimension
        metric='cosine',
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
chat_index = pc.Index(PINECONE_CHAT_INDEX)

# Index for file chunks
if PINECONE_VECTOR_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_VECTOR_INDEX,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
chunk_index = pc.Index(PINECONE_VECTOR_INDEX)

# Embedding Model
embedding_model = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

# LangChain Pinecone Vector Store (Correct method)
vector_db = LangchainPinecone.from_existing_index(
    index_name=PINECONE_VECTOR_INDEX,
    embedding=embedding_model,
    text_key="content"
)

# Retriever
retriever = vector_db.as_retriever()
