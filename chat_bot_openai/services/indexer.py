import os
import pdfplumber
import pytesseract
from PIL import Image
from fastapi import UploadFile
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.vectorstore import vector_db
from langchain_openai import OpenAIEmbeddings
from config.config import OPENAI_EMBEDDING_MODEL, UPLOAD_DIR

embedding_model = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
os.makedirs(UPLOAD_DIR, exist_ok=True)

def process_uploaded_files(files: list[UploadFile]):
    """Processes uploaded files and prepares them for ChromaDB indexing with correct format."""
    texts = []
    metadatas = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        extracted_text = ""
        if file.filename.endswith(".pdf"):
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
            continue

        chunks = splitter.split_text(extracted_text)
        texts.extend(chunks)
        metadatas.extend([{"source": file.filename}] * len(chunks))  # ✅ Consistent metadata

    if texts and metadatas:
        try:
            # Correct usage: separate texts and metadatas
            vector_db.add_texts(texts=texts, metadatas=metadatas)
            return {"message": f"Indexed {len(texts)} document chunks successfully!"}
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {str(e)}")
            return {"error": str(e)}
    else:
        return {"message": "No valid text found in uploaded files."}
