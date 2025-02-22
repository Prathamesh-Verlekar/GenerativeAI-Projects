import time
import os
from openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from services.vectorstore import retriever, chat_index
from datetime import datetime
from services.vectorstore import retriever, embedding_model 
from config.config import OPENAI_MODEL_NAME
from config.logging_config import logger

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model=OPENAI_MODEL_NAME, temperature=0.7, max_tokens=250)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=4)

qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# Store Chat History in Pinecone
def store_chat(session_id: str, question: str, answer: str):
    """Store chat history with embeddings in Pinecone."""
    embedding = embedding_model.embed_query(f"User: {question}\nAssistant: {answer}")
    pinecone_id = f"{session_id}_{datetime.utcnow().isoformat()}"
    chat_index.upsert(
        vectors=[(
            pinecone_id,
            embedding,
            {
                "session_id": session_id,
                "question": question, 
                "answer": answer       
            }
        )]
    )

def retrieve_history(session_id: str, top_k: int = 10):
    """Retrieve chat history for a given session from Pinecone."""
    results = chat_index.query(
        vector=embedding_model.embed_query(""),
        top_k=top_k,
        include_metadata=True,
        filter={"session_id": {"$eq": session_id}}  # Filter based on session_id
    )
    return [
        {"question": r.get("metadata", {}).get("question", ""), 
         "answer": r.get("metadata", {}).get("answer", "")}
        for r in results.get("matches", [])
    ]


def generate_answer(query: str, session_id: str):
    """Generate an answer using OpenAI's gpt-4o-mini and Pinecone retrieval, with a custom prompt."""
    logger.info(f"New Question Received: {query}")

    try:
        # Retrieve session history from Pinecone
        history = retrieve_history(session_id)
        context = "\n".join([f"User: {h['question']}\nAssistant: {h['answer']}" for h in history])

        # Custom Prompt
        prompt = f"""
        You are an intelligent AI assistant. Provide detailed, relevant, and helpful responses based on the context below.
        If the context does not provide sufficient information, rely on general knowledge to answer the query.
        If you truly have no knowledge of the answer, respond with: "I'm not sure about that."

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        # Generate response using Conversational RAG (Updated with 'invoke()' and prompt)
        start_time = time.time()
        response_dict = qa_chain.invoke({"question": prompt})
        end_time = time.time()
        logger.info(f"Response generated in {round((end_time - start_time) * 1000, 2)} ms.")

        # Extract 'answer' from the response
        cleaned_response = response_dict.get("answer", "I'm not sure about that.").strip()
        logger.info(f"Generated Response: {cleaned_response}")

        # Store chat in Pinecone for future context
        store_chat(session_id, query, cleaned_response)

        return {"session_id": session_id, "question": query, "answer": cleaned_response}

    except Exception as e:
        logger.error(f"Error Generating Answer: {str(e)}", exc_info=True)
        return {"error": "LLM failed to generate response"}

