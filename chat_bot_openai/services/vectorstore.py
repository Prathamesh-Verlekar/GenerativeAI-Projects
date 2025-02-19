from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from config.config import CHROMA_DB_PATH, OPENAI_EMBEDDING_MODEL

embedding_model = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

vector_db = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

retriever = vector_db.as_retriever()
