print(">>> start vectordb.py")
from chromadb import Client
from sentence_transformers import SentenceTransformer

client = Client()
print(">>> Chroma client initialised")

try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print(">>> Sentence-Transformer model loaded")
except Exception as e:
    print(">>> WARNING: embedding model failed to load —", e)
    embedder = None

try:
    collection = client.get_or_create_collection(
        name="invoices", embedding_function=None
    )
    print(">>> Chroma collection ready")
except Exception as e:
    print(">>> WARNING: could not create collection —", e)
    collection = None
