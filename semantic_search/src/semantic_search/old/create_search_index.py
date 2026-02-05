from os import getenv
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(getenv("MONGODB_URI"))
collection = client["RAG_Implementations"]["embedding_pdfs"]

search_index_model = SearchIndexModel(
  definition={
    "fields": [
      {
        "type": "vector",
     #    OpenAI number of dimensions below.
        "numDimensions": 1536,
        "path": "embeddings",
        "similarity": "dotProduct",
        "quantization": "scalar"
      }
    ]
  },
  name = "semantic_search_query_index",
  type = "vectorSearch",
)

collection.create_search_index(model = search_index_model)