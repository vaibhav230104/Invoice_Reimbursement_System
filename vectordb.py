print(">>> START vectordb.py")

from chromadb import Client
print(">>> chromadb import done")

from chromadb.config import Settings
print(">>> chromadb client settings done")

from sentence_transformers import SentenceTransformer
print(">>> sentence_transformers import done")

client = Client()
print(">>> chromadb client initialized")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
print(">>> embedder model loaded")

collection = client.get_or_create_collection(
    name="invoices",
    embedding_function=None
)
print(">>> collection initialized")
