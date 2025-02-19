import time
import os
from openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from services.vectorstore import retriever
from sqlalchemy.orm import Session
from models.chat_history import ChatHistory
from models.database import get_db
from config.config import OPENAI_MODEL_NAME
from config.logging_config import logger

# OpenAI Configuration
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize OpenAI Chat Model (Using v1/chat/completions endpoint)
logger.info("Initializing OpenAI gpt-4o-mini chat model...")
llm = ChatOpenAI(
    model=OPENAI_MODEL_NAME,  # gpt-4o-mini
    temperature=0.7,
    max_tokens=250
)
logger.info("OpenAI Chat Model Initialized Successfully.")

# Initialize Memory for Conversation History
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    k=4  # Store last 4 interactions
)

# Create Conversational RAG Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# Function to Fetch Last 4 Chat History from DB
def get_last_4_chats(db: Session):
    return db.query(ChatHistory).order_by(ChatHistory.id.desc()).limit(4).all()

# Function to Generate Answer with Contextual History
def generate_answer(query: str, db_session: Session = None):
    """
    Generate an answer using OpenAI's gpt-4o-mini chat model while incorporating the last 4 chat interactions.
    """
    logger.info(f"New Question Received: {query}")

    try:
        # Ensure database session is provided
        if db_session is None:
            db_session = next(get_db())

        chat_history = get_last_4_chats(db_session)

        # Format chat history as a string
        history_text = "\n".join(
            [f"User: {chat.question}\nAssistant: {chat.answer}" for chat in reversed(chat_history)]
        )

        # Include chat history in the query
        full_query = f"Chat History:\n{history_text}\n\nNew Question: {query}"

        # Start timing LLM inference
        start_time = time.time()

        # Generate response using Conversational RAG
        raw_response = qa_chain.run({"question": full_query, "chat_history": chat_history})

        # Capture end time after LLM completes response generation
        end_time = time.time()
        response_time = round((end_time - start_time) * 1000, 2)
        logger.info(f"Response generated in {response_time} ms.")

        # Check the type of response and clean it
        cleaned_response = str(raw_response).strip()
        logger.info(f"Generated Response: {cleaned_response}")

        # Store new conversation in the database
        new_chat = ChatHistory(question=query, answer=cleaned_response)
        db_session.add(new_chat)
        db_session.commit()

        return {"question": query, "answer": cleaned_response}

    except Exception as e:
        logger.error(f"Error Generating Answer: {str(e)}", exc_info=True)
        return {"error": "LLM failed to generate response"}


