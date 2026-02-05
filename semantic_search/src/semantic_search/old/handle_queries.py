from os import getenv
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
query = input("What would you like to find in the database?")
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
query_vector = embeddings.embed_query(query)
print(query_vector)