# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from langchain.chains import RetrievalQA
# from langchain_community.llms import HuggingFacePipeline
# from services.vectorstore import retriever
# from utils.prompts import custom_prompt
# from config import MODEL_NAME


# #Detect Device (Use GPU if available, else CPU)
# device = "mps" if torch.cuda.is_available() else "cpu"

# #Load Tokenizer & Model (Optimized for TinyLlama)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     device_map="auto",  #Let Hugging Face automatically assign the best device
#     torch_dtype=torch.float16 if device == "mps" else torch.float32,  #float16 for GPU, float32 for CPU
#     low_cpu_mem_usage=True  #Reduce memory usage
# )

# #Optimized HF Pipeline for Speed
# generator_pipeline = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=200,  #Reduce max tokens for faster responses
#     do_sample=True,
#     temperature=0.7,
#     top_k=50,
#     top_p=0.9,
#     repetition_penalty=1.1,
#     return_full_text=False
# )

# #Wrap in LangChain’s HuggingFacePipeline
# llm = HuggingFacePipeline(pipeline=generator_pipeline)

# #Create RetrievalQA
# qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": custom_prompt})

# # Function to generate answers
# def generate_answer(query):
#     raw_response = qa_chain.run(query)

#     # Post-process response to remove extra formatting
#     cleaned_response = (
#         raw_response.strip()
#         .replace("Context:", "")
#         .replace("Question:", "")
#         .replace("Answer:", "")
#         .replace("\n", " ")
#         .replace("<s>", "")
#         .replace("</s>", "")
#         .strip()
#     )

#     return cleaned_response

import time
import torch
from fastapi import Depends
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from services.vectorstore import retriever
from utils.prompts import custom_prompt
from config.config import MODEL_NAME
from models.database import get_db  # Import DB session
from models.chat_history import ChatHistory  # Chat history model
from sqlalchemy.orm import Session
from config.logging_config import logger

# Detect Device (Use GPU if available, else CPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load Tokenizer & Model
logger.info("Loading LLM model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if device == "mps" else torch.float32,
    low_cpu_mem_usage=True
)

logger.info("Model Loaded Successfully")

# Optimized HF Pipeline
generator_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=250,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
    return_full_text=False
)

# Wrap in LangChain’s HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=generator_pipeline)

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
    Generate an answer using the LLM while incorporating the last 4 chat interactions.
    """
    logger.info(f"New Question Received: {query}")

    try:
        # Ensure database session is provided
        if db_session is None:
            with next(get_db()) as db_session:  # Fetch database session
                chat_history = get_last_4_chats(db_session)
        else:
            chat_history = get_last_4_chats(db_session)

        # Format chat history as a string
        history_text = "\n".join([f"User: {chat.question}\nAssistant: {chat.answer}" for chat in reversed(chat_history)])

        # Include chat history in the query
        full_query = f"Chat History:\n{history_text}\n\nNew Question: {query}"

        # Start timing LLM inference
        start_time = time.time()

        # Generate response using Conversational RAG
        raw_response = qa_chain.run({"question": full_query, "chat_history": chat_history})

        # Capture end time after LLM completes response generation
        end_time = time.time()
        response_time = round((end_time - start_time) * 1000, 2)
        print(response_time)

        # Ensure the response is only a string, not a dictionary
        if isinstance(raw_response, dict) and "answer" in raw_response:
            cleaned_response = raw_response["answer"]  # Extract only the answer text
            logger.info(f"Generated Response: {cleaned_response}")
        elif isinstance(raw_response, dict):
            cleaned_response = str(raw_response)  # Convert entire dict to string if necessary
            logger.info(f"Generated Response: {cleaned_response}")
        else:
            cleaned_response = str(raw_response).strip()  # Ensure response is a string
            logger.info(f"Generated Response: {cleaned_response}")

        # Store new conversation in the database
        new_chat = ChatHistory(question=query, answer=cleaned_response)  # Store only string
        db_session.add(new_chat)
        db_session.commit()

        return {"question": query, "answer": cleaned_response}
    
    except Exception as e:
        logger.error(f"Error Generating Answer: {str(e)}", exc_info=True)
        return {"error": "LLM failed to generate response"}


