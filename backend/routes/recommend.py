from fastapi import APIRouter, Depends
from pydantic import BaseModel
import numpy as np
import time
from typing import List, Optional
from backend.faiss_index import search_candidates, EMBEDDING_DIM
from backend.redis_client import get_vibe_studio_state, set_vibe_studio_state
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from backend.database import async_session, Track

router = APIRouter()

class RecommendRequest(BaseModel):
    user_id: int
    limit: int = 20
    apply_vibe_steering: bool = True

class VibeState(BaseModel):
    user_id: int
    energy: float = 0.5
    valence: float = 0.5
    acousticness: float = 0.5
    danceability: float = 0.5

async def get_db():
    async with async_session() as session:
        yield session

def compute_ild(embeddings: List[List[float]]) -> float:
    if not embeddings or len(embeddings) < 2: return 0.0
    mat = np.array(embeddings)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / (norms + 1e-8)
    sim_matrix = np.dot(mat, mat.T)
    # Average pairwise distance (1 - similarity) for off-diagonal
    n = len(mat)
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    distances = 1.0 - sim_matrix[mask]
    return float(np.mean(distances))

def compute_novelty(tracks: List[Track]) -> float:
    # Mock novelty score: simulated based on internal ID (higher ID = less popular/more novel in our mock setup)
    if not tracks: return 0.0
    return float(np.mean([(t.id % 10) / 10.0 for t in tracks]))

@router.post("/vibe")
async def update_vibe(vibe: VibeState):
    await set_vibe_studio_state(str(vibe.user_id), vibe.model_dump())
    return {"status": "updated"}

@router.post("/recommend")
async def get_recommendations(req: RecommendRequest, db: AsyncSession = Depends(get_db)):
    pipeline_start = time.perf_counter()
    
    # 1. Fetch Vibe State
    t0 = time.perf_counter()
    vibe = await get_vibe_studio_state(str(req.user_id))
    vibe_fetch_ms = (time.perf_counter() - t0) * 1000
    
    # 2. Compute query vector
    user_vector = np.random.randn(1, EMBEDDING_DIM).astype(np.float32)
    
    if req.apply_vibe_steering and vibe:
        energy = float(vibe.get("energy", 0.5))
        valence = float(vibe.get("valence", 0.5))
        acousticness = float(vibe.get("acousticness", 0.5))
        danceability = float(vibe.get("danceability", 0.5))
        
        # Simple projection heuristic
        vibe_modifier = np.zeros((1, EMBEDDING_DIM))
        vibe_modifier[0, 0:32] = energy - 0.5
        vibe_modifier[0, 32:64] = valence - 0.5
        vibe_modifier[0, 64:96] = danceability - 0.5
        vibe_modifier[0, 96:128] = 0.5 - acousticness
        
        query_vector = user_vector + 0.5 * vibe_modifier.astype(np.float32)
    else:
        query_vector = user_vector

    query_vector = query_vector / np.linalg.norm(query_vector)

    # 3. Retrieve from FAISS
    t0 = time.perf_counter()
    distances, candidate_faiss_ids = search_candidates(query_vector, top_k=req.limit * 5)
    faiss_ms = (time.perf_counter() - t0) * 1000
    candidates_before_dedup = len(candidate_faiss_ids)
    
    # 4. DB Fetch
    t0 = time.perf_counter()
    if candidate_faiss_ids:
        result = await db.execute(select(Track).where(Track.id.in_(candidate_faiss_ids)))
        tracks_unsorted = {t.id: t for t in result.scalars().all()}
        # Reorder to match FAISS output
        tracks = [tracks_unsorted[tid] for tid in candidate_faiss_ids if tid in tracks_unsorted]
    else:
        tracks = []
    db_ms = (time.perf_counter() - t0) * 1000
    
    # 5. Mock Ranking
    t0 = time.perf_counter()
    # In production: Deep & Wide model scoring here
    # We mock it by adding a slight random noise to distances and re-sorting
    ranking_scores = [d + np.random.uniform(-0.1, 0.1) for d in (distances[0] if len(distances)>0 else [])]
    ranked_indices = np.argsort(ranking_scores)
    ranked_tracks = [tracks[i] for i in ranked_indices if i < len(tracks)]
    ranking_ms = (time.perf_counter() - t0) * 1000

    # 6. Post-Processing & Deduplication
    t0 = time.perf_counter()
    unique_artists = set()
    final_tracks = []
    
    for t in ranked_tracks:
        if len(final_tracks) >= req.limit:
            break
        # Simple artist dedup: max 2 tracks per artist
        if t.artist not in unique_artists or len([f for f in final_tracks if f.artist == t.artist]) < 2:
            unique_artists.add(t.artist)
            final_tracks.append(t)
            
    postproc_ms = (time.perf_counter() - t0) * 1000
    
    total_ms = (time.perf_counter() - pipeline_start) * 1000
    
    # Metrics calculation
    embeddings = [t.embedding for t in final_tracks if t.embedding]
    ild = compute_ild(embeddings)
    novelty = compute_novelty(final_tracks)
    genre_spread = len({t.genre for t in final_tracks if t.genre})
    
    response_tracks = [{
        "id": t.id,
        "title": t.title,
        "artist": t.artist,
        "genre": t.genre,
        "energy": t.energy,
        "valence": t.valence,
        "bpm": t.bpm
    } for t in final_tracks]

    return {
        "user_id": req.user_id,
        "tracks": response_tracks,
        "pipeline_metrics": {
            "vibe_fetch_ms": round(vibe_fetch_ms, 2),
            "faiss_search_ms": round(faiss_ms, 2),
            "db_fetch_ms": round(db_ms, 2),
            "ranking_ms": round(ranking_ms, 2),
            "postprocessing_ms": round(postproc_ms, 2),
            "total_pipeline_ms": round(total_ms, 2),
        },
        "quality_metrics": {
            "intra_list_diversity": round(ild, 4),
            "novelty_score": round(novelty, 4),
            "genre_spread": genre_spread,
            "unique_artists": len(unique_artists),
            "candidates_before_dedup": candidates_before_dedup,
            "candidates_after_dedup": len(final_tracks),
        },
        "vibe_state": vibe,
        "query_vector_preview": query_vector[0][:8].tolist(),
        "vibe_steering_applied": bool(vibe and req.apply_vibe_steering)
    }
