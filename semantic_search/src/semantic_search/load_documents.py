from os import getenv
from glob import glob
from uuid import uuid4
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(getenv("MONGODB_URI"))
collection = client["RAG_Implementations"]["embedding_pdfs"]
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, add_start_index = True)
for pdf in glob(getenv("PDF_PATH")):
     pdf_pages = PyPDFLoader(pdf).load()
     pdf_chunks = text_splitter.split_documents(pdf_pages)
     for pdf_chunk in pdf_chunks:
          pdf_chunk.id = str(uuid4())
          collection.insert_one(pdf_chunk.model_dump())


