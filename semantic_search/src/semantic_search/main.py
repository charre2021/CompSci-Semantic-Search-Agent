import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

docs = PyPDFLoader(os.getenv("PDF_PATH")).load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, add_start_index = True)
all_splits = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")
test_vector = embeddings.embed_query(all_splits[0].page_content)
print(test_vector[:10])


