import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import json
from typing import List, Dict

# Global clients
chroma_client = None
collection = None
embedder = None

def init_rag():
    global chroma_client, collection, embedder
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device="cpu"
    )
    
    collection = chroma_client.get_or_create_collection(
        name="sidh_guide",
        embedding_function=ef
    )

def ingest_data():
    with open('data/courses.json', 'r') as f:
        docs = json.load(f)
    
    texts = [doc['content'] for doc in docs]
    metadatas = [{"title": doc['title'], "id": doc['id']} for doc in docs]
    ids = [doc['id'] for doc in docs]
    
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Ingested {len(ids)} documents")

def query_rag(question: str, k: int = 1) -> List[Dict]:
    question_embedding = embedder.encode([question])
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=k,
        include=['documents', 'metadatas']
    )
    return [{"text": doc, "metadata": meta} for doc, meta in zip(results['documents'][0], results['metadatas'][0])]

if __name__ == "__main__":
    init_rag()
    ingest_data()
