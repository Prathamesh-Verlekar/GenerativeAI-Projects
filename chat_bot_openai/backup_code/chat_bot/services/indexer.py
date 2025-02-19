import os
import pdfplumber
import pytesseract
from PIL import Image
from fastapi import UploadFile
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.vectorstore import vector_db

UPLOAD_DIR = "uploaded_files"

# Ensure Upload Directory Exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

def process_uploaded_files(files: list[UploadFile]):
    """Processes uploaded PDFs, images, and text files into ChromaDB."""

    documents = []
    
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # Save Uploaded File
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        extracted_text = ""

        # Extract Text from PDFs
        if file.filename.endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                extracted_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

        # Extract Text from Images using OCR
        elif file.filename.endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(file_path)
            extracted_text = pytesseract.image_to_string(img).strip()  # Strip empty spaces

        # Extract Text from Text Files
        elif file.filename.endswith((".txt", ".md")):
            loader = TextLoader(file_path)
            text_docs = loader.load()
            extracted_text = "\n".join([doc.page_content for doc in text_docs])

        # If No Text Found, Skip Adding to Database
        if not extracted_text.strip():  # Ensure no empty text
            print(f"⚠️ Skipping {file.filename}: No extractable text found.")
            continue

        documents.append(extracted_text)

    #Split Large Documents for Efficient Search
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
    split_docs = text_splitter.create_documents(documents)

    #Ensure at least 1 document exists before adding to ChromaDB
    if split_docs:
        vector_db.add_documents(split_docs)
        return {"message": f"Indexed {len(split_docs)} document chunks successfully!"}
    else:
        return {"message": "⚠️ No valid text found in uploaded files."}

# import os
# import pdfplumber
# import pytesseract
# from PIL import Image
# from fastapi import UploadFile
# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from services.vectorstore import vector_db, create_bm25_retriever

# UPLOAD_DIR = "uploaded_files"
# os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure Upload Directory Exists

# def process_uploaded_files(files: list[UploadFile]):
#     """Processes uploaded PDFs, images, and text files into ChromaDB."""

#     documents = []
    
#     for file in files:
#         file_path = os.path.join(UPLOAD_DIR, file.filename)

#         # Save Uploaded File
#         with open(file_path, "wb") as buffer:
#             buffer.write(file.file.read())

#         extracted_text = ""

#         # Extract Text from PDFs
#         if file.filename.endswith(".pdf"):
#             with pdfplumber.open(file_path) as pdf:
#                 extracted_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

#         # Extract Text from Images using OCR
#         elif file.filename.endswith((".png", ".jpg", ".jpeg")):
#             img = Image.open(file_path)
#             extracted_text = pytesseract.image_to_string(img).strip()

#         # Extract Text from Text Files
#         elif file.filename.endswith((".txt", ".md")):
#             loader = TextLoader(file_path)
#             text_docs = loader.load()
#             extracted_text = "\n".join([doc.page_content for doc in text_docs])

#         # Skip Empty Files
#         if not extracted_text.strip():
#             print(f"⚠️ Skipping {file.filename}: No extractable text found.")
#             continue

#         documents.append(extracted_text)

#     # Split Large Documents for Efficient Search
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
#     split_docs = text_splitter.create_documents(documents)

#     # Ensure at least 1 document exists before adding to ChromaDB
#     if split_docs:
#         vector_db.add_documents(split_docs)  # Add to ChromaDB

#         # Update BM25 retriever after indexing new documents
#         global bm25_retriever
#         bm25_retriever = create_bm25_retriever()

#         return {"message": f"Indexed {len(split_docs)} document chunks successfully!"}
#     else:
#         return {"message": "⚠️ No valid text found in uploaded files."}

