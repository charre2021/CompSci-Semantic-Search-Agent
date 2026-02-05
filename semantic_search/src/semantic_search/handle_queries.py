from os import getenv
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
query = input("What would you like to find in the database? ")
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
query_vector = embeddings.embed_query(query)
client = MongoClient(getenv("MONGODB_URI"))
vector_store = MongoDBAtlasVectorSearch.from_connection_string(
     connection_string = getenv("MONGODB_URI"),
     namespace = "semantic_search.comp_sci_pdfs",
     embedding = embeddings,
     index_name= "vector_index",
)

results = vector_store.similarity_search_by_vector(query_vector)
print(results[:10])