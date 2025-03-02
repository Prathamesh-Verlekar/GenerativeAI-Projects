from openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from services.vectorstore import retriever, chat_index, chunk_index
from datetime import datetime
from services.vectorstore import retriever, embedding_model 
from utils.prompts import prompt as default_prompt
from config.config import OPENAI_MODEL_NAME
from config.logging_config import logger
from sklearn.metrics.pairwise import cosine_similarity
import logging
import numpy as np
import time
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model=OPENAI_MODEL_NAME, temperature=0.7, max_tokens=500)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=4)

qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# Store Chat History in Pinecone
def store_chat(session_id: str, question: str, answer: str):
    """Store chat history with embeddings in Pinecone, ensuring correct session ID association."""
    embedding = embedding_model.embed_query(f"User: {question}\nAssistant: {answer}")
    pinecone_id = f"{session_id}_{datetime.utcnow().isoformat()}"
    chat_index.upsert(
        vectors=[{
            "id": pinecone_id,
            "values": embedding,
            "metadata": {
                "session_id": session_id,
                "question": question,
                "answer": answer       
            }
        }]
    )

# Determine relevance of first question with second
def is_relevant(current_query_embedding, history_embeddings, threshold=0.7):
    """Check if current query is relevant to history based on cosine similarity."""
    if not history_embeddings:
        return False
    similarities = cosine_similarity([current_query_embedding], history_embeddings)
    max_similarity = np.max(similarities) if similarities.size else 0
    logger.info(f"Max similarity score with history: {max_similarity}")
    return max_similarity >= threshold

def retrieve_history(session_id: str, top_k: int = 4):
    """Retrieve last 4 chat history messages for a given session from Pinecone."""
    logging.info(f"Querying Pinecone for chat history of session_id: {session_id}")
    results = chat_index.query(
        vector=[0] * 1536,
        top_k=top_k,
        include_metadata=True,
        filter={"session_id": {"$eq": session_id}}
    )
    if not results.get("matches"):
        logging.warning(f"No chat history found in Pinecone for session_id: {session_id}")
        return []
    return [
        {"question": r["metadata"].get("question", ""), "answer": r["metadata"].get("answer", "")}
        for r in results.get("matches", []) if "metadata" in r
    ]

def retrieve_relevant_chunks(session_id: str, query: str, top_k: int = 5):
    """Retrieve relevant document chunks from Pinecone based on query similarity."""
    logging.info(f"Searching for relevant chunks for session_id: {session_id}")
    query_embedding = embedding_model.embed_query(query)
    results = chunk_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter={"session_id": {"$eq": session_id}}
    )
    if not results.get("matches"):
        logging.warning(f"No matching documents found in Pinecone for session_id: {session_id}")
        return []
    return [{"content": r["metadata"].get("content", "")} for r in results.get("matches", []) if "metadata" in r]


def generate_answer(query: str, session_id: str):
    """Generate an answer using OpenAI's gpt-4o-mini and Pinecone retrieval, with a custom prompt."""
    logger.info(f"New Question Received: {query}")

    try:
        relevant_chunks = retrieve_relevant_chunks(session_id, query)
        context = "\n".join([chunk["content"] for chunk in relevant_chunks]) if relevant_chunks else "No relevant context found."
        # Retrieve session history
        history = retrieve_history(session_id)
        history_texts = [f"User: {h['question']}\nAssistant: {h['answer']}" for h in history]

        # Embed current query
        current_query_embedding = embedding_model.embed_query(query)
        history_embeddings = [embedding_model.embed_query(h["question"]) for h in history]

        # Check relevance
        relevant = is_relevant(current_query_embedding, history_embeddings)
        history_context = "\n".join([f"User: {h['question']}\nAssistant: {h['answer']}" for h in history]) if relevant else ""
        if not relevant:
            logger.info("No relevant history found. Proceeding without history context.")

        # Generate response using Conversational RAG (Updated with 'invoke()' and prompt)
        start_time = time.time()
        full_context = f"Relevant Documents:\n{context}\n\nChat History:\n{history_context}"
        final_prompt = default_prompt.format(context=full_context, query=query)
        # Conditionally generate the answer:
        if relevant:
            # Use qa_chain when history is relevant
            response_dict = qa_chain.invoke({"question": final_prompt})
        else:
            # Direct LLM call when history is irrelevant
            response_dict = llm.predict(final_prompt)

        # Handle different response structures
        cleaned_response = (
            response_dict.get("answer", response_dict) if isinstance(response_dict, dict) else response_dict
        ).strip()

        if not cleaned_response:
            cleaned_response = "I'm not sure about that."

        logger.info(f"Generated Response: {cleaned_response}")

        # Store chat in Pinecone
        store_chat(session_id, query, cleaned_response)
        return {"session_id": session_id, "question": query, "answer": cleaned_response}

    except Exception as e:
        logger.error(f"Error Generating Answer: {str(e)}", exc_info=True)
        return {"error": "LLM failed to generate response"}

