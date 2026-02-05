from os import getenv
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(getenv("MONGODB_URI"))
collection = client["RAG_Implementations"]["embedding_pdfs"]
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
for document in collection.find():
     embedding = embeddings.embed_query(document['page_content'])
     collection.update_one({ '_id': document['_id'] }, { "$set": { 'embeddings': embedding } }, upsert = True)


