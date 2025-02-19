from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.config import CHROMA_DB_PATH, EMBEDDING_MODEL

# Load Embeddings
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Initialize ChromaDB
vector_db = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine", "hnsw:ef_construction": 200}
)

#Return retriever instead of function
retriever = vector_db.as_retriever()

# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever
# from langchain.docstore.document import Document
# from config import CHROMA_DB_PATH, EMBEDDING_MODEL

# # Load Embeddings Model
# embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# # Initialize ChromaDB for Dense Vector Search
# vector_db = Chroma(
#     persist_directory=CHROMA_DB_PATH,
#     embedding_function=embedding_model,
#     collection_metadata={"hnsw:space": "cosine", "hnsw:ef_construction": 200}
# )
# vector_retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# # Create BM25 Retriever for Keyword Matching
# def create_bm25_retriever():
#     docs = vector_db.get()  # Fetch documents

#     if not docs or "ids" not in docs:  # Handle empty ChromaDB
#         print("⚠️ No documents found in ChromaDB for BM25.")
#         return None

#     # Convert the retrieved docs to LangChain Document format
#     documents = [Document(page_content=text) for text in docs["documents"]]  # Correct way to extract text

#     # Initialize BM25 Retriever
#     bm25_retriever = BM25Retriever.from_documents(documents)
#     bm25_retriever.k = 10  # Retrieve top 10 keyword matches
#     return bm25_retriever

# bm25_retriever = create_bm25_retriever()

# # Combine BM25 + ChromaDB using EnsembleRetriever
# if bm25_retriever:
#     hybrid_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5])  
# else:
#     hybrid_retriever = vector_retriever  # Use Vector Retriever if BM25 is empty

# retriever = hybrid_retriever  # Use Hybrid Retriever in RAG

