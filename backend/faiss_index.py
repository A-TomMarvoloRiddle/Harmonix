import faiss
import numpy as np

EMBEDDING_DIM = 128

# Using IndexFlatIP for Inner Product (cosine similarity if normalized)
# Alternatively, IndexHNSWFlat for faster approximate nearest neighbors
# Initializing HNSW index for 128-dimensional dense audio embeddings
index = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32)
index.hnsw.efSearch = 64
index.hnsw.efConstruction = 64

def add_embeddings(embeddings: np.ndarray):
    """Add 128-d embeddings to the FAISS index."""
    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Embeddings must be {EMBEDDING_DIM}-dimensional")
    index.add(embeddings.astype(np.float32))

def search_candidates(query_vector: np.ndarray, top_k: int = 500):
    """Retrieve top-k candidates based on vector similarity."""
    distances, indices = index.search(query_vector.astype(np.float32), top_k)
    return distances, indices
