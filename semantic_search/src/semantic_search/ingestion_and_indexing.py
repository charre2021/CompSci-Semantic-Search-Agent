from os import getenv
from glob import glob
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(getenv("MONGODB_URI"))
collection = client["semantic_search"]["comp_sci_pdfs"]
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, add_start_index = True)
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
vector_store = MongoDBAtlasVectorSearch.from_connection_string(
     connection_string = getenv("MONGODB_URI"),
     namespace = "semantic_search.comp_sci_pdfs",
     embedding = embeddings,
     index_name= "vector_index",
)

for pdf in glob(getenv("PDF_PATH")):
     pdf_pages = PyPDFLoader(pdf).load()
     pdf_chunks = text_splitter.split_documents(pdf_pages)
     vector_store.add_documents(pdf_chunks)
     
vector_store.create_vector_search_index(dimensions = 1536)

