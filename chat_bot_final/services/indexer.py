import os
import pandas as pd
import pdfplumber
import pytesseract
from PIL import Image
from fastapi import UploadFile
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.vectorstore import vector_db, chunk_index, embedding_model  # Import embedding_model from vectorstore
from datetime import datetime
from config.config import UPLOAD_DIR
from config.logging_config import logger  # Logging for debugging

# Ensure Upload Directory Exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

def process_uploaded_files(files: list[UploadFile], session_id: str):
    """
    Processes uploaded files (CSV, Excel, PDF, Image, Text) and indexes them into Pinecone
    with embeddings and session-based metadata.
    """
    texts = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=100)

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        extracted_text = ""

        try:
            # Extract Text Based on File Type
            if file.filename.endswith(".csv"):
                df = pd.read_csv(file_path)
                extracted_text = df.to_string(index=False)

            elif file.filename.endswith((".xls", ".xlsx")):
                df = pd.read_excel(file_path)
                extracted_text = df.to_string(index=False)

            elif file.filename.endswith(".pdf"):
                with pdfplumber.open(file_path) as pdf:
                    extracted_text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])

            elif file.filename.endswith((".png", ".jpg", ".jpeg")):
                img = Image.open(file_path)
                extracted_text = pytesseract.image_to_string(img).strip()

            elif file.filename.endswith((".txt", ".md")):
                loader = TextLoader(file_path)
                text_docs = loader.load()
                extracted_text = "\n".join([doc.page_content for doc in text_docs])

            if not extracted_text.strip():
                logger.warning(f"No valid text extracted from {file.filename}")
                continue

            # Split and Embed Text
            chunks = splitter.split_text(extracted_text)
            embeddings = embedding_model.embed_documents(chunks)  # OpenAI embedding

            for chunk_text, embedding in zip(chunks, embeddings):
                pinecone_id = f"{session_id}_{datetime.utcnow().isoformat()}"
                texts.append(
                    (pinecone_id, embedding, {
                        "session_id": session_id,
                        "source": file.filename,
                        "content": chunk_text
                    })
                )

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")

    # Indexing Data into Pinecone
    if texts:
        try:
            chunk_index.upsert(vectors=texts)
            logger.info(f"Indexed {len(texts)} document chunks for session {session_id}.")
            return {"message": f"Indexed {len(texts)} document chunks successfully!"}
        except Exception as e:
            logger.error(f"Error adding documents to Pinecone: {str(e)}")
            return {"error": str(e)}
    else:
        logger.warning("No valid text found in uploaded files.")
        return {"message": "No valid text found in uploaded files."}



