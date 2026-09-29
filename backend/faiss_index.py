import faiss
import numpy as np

EMBEDDING_DIM = 128

# Using IndexFlatIP for Inner Product (cosine similarity if normalized)
# Alternatively, IndexHNSWFlat for faster approximate nearest neighbors
# Initializing HNSW index for 128-dimensional dense audio embeddings
index = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32)
index.hnsw.efSearch = 64
index.hnsw.efConstruction = 64

_track_id_map = []  # maps faiss_idx → db_track_id

def add_embeddings(embeddings: np.ndarray, track_ids: list):
    """Add 128-d embeddings to the FAISS index with track IDs."""
    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Embeddings must be {EMBEDDING_DIM}-dimensional")
    index.add(embeddings.astype(np.float32))
    _track_id_map.extend(track_ids)

def search_candidates(query_vector: np.ndarray, top_k: int = 500):
    """Retrieve top-k candidates based on vector similarity, returning DB track IDs."""
    distances, indices = index.search(query_vector.astype(np.float32), top_k)
    
    # Map back to real DB IDs
    db_ids = []
    for i in indices[0]:
        if i != -1 and i < len(_track_id_map):
            db_ids.append(_track_id_map[i])
            
    return distances, db_ids

def get_all_embeddings():
    """Return all stored embeddings and their track_ids for t-SNE."""
    if not _track_id_map:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), []
    n = index.ntotal
    embeddings = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
    for i in range(n):
        try:
            embeddings[i] = index.reconstruct(i)
        except Exception:
            pass # fallback if reconstruct fails, but HNSWFlat supports reconstruct
    return embeddings, list(_track_id_map)

def get_index_size():
    return index.ntotal

async def warm_up_faiss():
    """Load all embeddings from the database into the FAISS index."""
    import numpy as np
    from backend.database import async_session, Track
    from sqlalchemy.future import select
    
    global _track_id_map
    
    async with async_session() as session:
        result = await session.execute(select(Track.id, Track.embedding))
        tracks = result.all()
        
    embeddings = []
    track_ids = []
    
    for t in tracks:
        if t.embedding is not None:
            embeddings.append(t.embedding)
            track_ids.append(t.id)
            
    if embeddings:
        emb_matrix = np.array(embeddings, dtype=np.float32)
        # Reset index and map
        index.reset()
        _track_id_map.clear()
        
        # Add to index
        add_embeddings(emb_matrix, track_ids)
        print(f"FAISS index warmed up with {len(track_ids)} tracks.")
    else:
        print("No embeddings found in DB to warm up FAISS.")

