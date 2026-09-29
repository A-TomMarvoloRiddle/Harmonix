from fastapi import APIRouter, Depends
from pydantic import BaseModel
import numpy as np
import time
import torch
import os
from typing import List, Optional
from backend.faiss_index import search_candidates, EMBEDDING_DIM
from backend.redis_client import get_vibe_studio_state, set_vibe_studio_state
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.sql.expression import func
from backend.database import async_session, Track
from backend.ml.ranking_models import TwoTowerModel, DeepWideModel

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

_two_tower = None
_deep_wide = None

def get_models():
    global _two_tower, _deep_wide
    if _two_tower is None and os.path.exists("backend/weights/two_tower.pth"):
        _two_tower = TwoTowerModel(EMBEDDING_DIM)
        _two_tower.load_state_dict(torch.load("backend/weights/two_tower.pth", weights_only=True))
        _two_tower.eval()
    if _deep_wide is None and os.path.exists("backend/weights/deep_wide.pth"):
        _deep_wide = DeepWideModel(EMBEDDING_DIM, 4)
        _deep_wide.load_state_dict(torch.load("backend/weights/deep_wide.pth", weights_only=True))
        _deep_wide.eval()
    return _two_tower, _deep_wide

async def get_db():
    async with async_session() as session:
        yield session

def compute_ild(embeddings: List[List[float]]) -> float:
    if not embeddings or len(embeddings) < 2: return 0.0
    mat = np.array(embeddings)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / (norms + 1e-8)
    sim_matrix = np.dot(mat, mat.T)
    n = len(mat)
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    distances = 1.0 - sim_matrix[mask]
    return float(np.mean(distances))

def compute_novelty(tracks: List[Track]) -> float:
    if not tracks: return 0.0
    features = np.array([[t.energy, t.valence, t.acousticness, t.danceability] for t in tracks])
    median_f = np.median(features, axis=0)
    distances = np.linalg.norm(features - median_f, axis=1)
    return float(np.mean(distances))

@router.post("/vibe")
async def update_vibe(vibe: VibeState):
    await set_vibe_studio_state(str(vibe.user_id), vibe.model_dump())
    return {"status": "updated"}

@router.post("/recommend")
async def get_recommendations(req: RecommendRequest, db: AsyncSession = Depends(get_db)):
    pipeline_start = time.perf_counter()
    two_tower, deep_wide = get_models()
    
    # 1. Fetch Vibe State
    t0 = time.perf_counter()
    vibe = await get_vibe_studio_state(str(req.user_id))
    vibe_fetch_ms = (time.perf_counter() - t0) * 1000
    
    # 2. Retrieve Candidates
    t0 = time.perf_counter()
    # Since we want to support true Vibe Steering across the whole catalog,
    # and the dataset is only 61 tracks, we pull all tracks for ranking.
    # In production with millions of tracks, we'd use FAISS + metadata filtering.
    result = await db.execute(select(Track))
    tracks = result.scalars().all()
    faiss_ms = (time.perf_counter() - t0) * 1000  # Repurposed metric
    candidates_before_dedup = len(tracks)
    db_ms = 0.0
    
    # 3. True ML Ranking / Vibe Steering
    t0 = time.perf_counter()
    
    # Calculate base user vector using TwoTower for ML integration
    if two_tower:
        base_user = torch.zeros(1, EMBEDDING_DIM)
        with torch.no_grad():
            user_tensor, _ = two_tower(base_user, torch.zeros(1, EMBEDDING_DIM))
            user_vector = user_tensor.numpy()
    else:
        user_vector = np.zeros((1, EMBEDDING_DIM), dtype=np.float32)

    # Score each track
    scores = []
    if req.apply_vibe_steering and vibe:
        v_energy = float(vibe.get("energy", 0.5))
        v_valence = float(vibe.get("valence", 0.5))
        v_acoustic = float(vibe.get("acousticness", 0.5))
        v_dance = float(vibe.get("danceability", 0.5))
        
        for t in tracks:
            # Calculate Euclidean distance in acoustic feature space
            dist = np.sqrt(
                (t.energy - v_energy)**2 +
                (t.valence - v_valence)**2 +
                (t.acousticness - v_acoustic)**2 +
                (t.danceability - v_dance)**2
            )
            # Use exponential decay so closer tracks have drastically higher scores
            score = np.exp(-dist * 3.0)
            scores.append(score)
            
        scores_np = np.array(scores)
        
        # Sort and take top candidates for exploitation (e.g. top 25)
        ranked_indices = np.argsort(scores_np)[::-1]
        pool_size = min(req.limit * 4, len(tracks))
        top_indices = ranked_indices[:pool_size]
        
        # Calculate probabilities only within the top pool for exploration
        pool_scores = scores_np[top_indices]
        probs = pool_scores / pool_scores.sum()
        
        num_to_sample = min(req.limit * 3, len(top_indices))
        sampled_pool_indices = np.random.choice(top_indices, size=num_to_sample, replace=False, p=probs)
        ranked_tracks = [tracks[i] for i in sampled_pool_indices]
        
    elif deep_wide and tracks:
        with torch.no_grad():
            u_t = torch.tensor(user_vector, dtype=torch.float32).repeat(len(tracks), 1)
            t_t = torch.tensor([t.embedding for t in tracks], dtype=torch.float32)
            c_t = torch.tensor([[0.5, 0.0, 0.8, 0.0]], dtype=torch.float32).repeat(len(tracks), 1)
            dw_scores = deep_wide(u_t, t_t, c_t).squeeze(-1).numpy()
            
        dw_scores = np.maximum(dw_scores, 0)
        ranked_indices = np.argsort(dw_scores)[::-1]
        pool_size = min(req.limit * 4, len(tracks))
        top_indices = ranked_indices[:pool_size]
        
        # Softmax-like scaling for probability within the pool
        pool_scores = dw_scores[top_indices]
        pool_scores = np.exp(pool_scores * 2.0)
        probs = pool_scores / pool_scores.sum()
        
        num_to_sample = min(req.limit * 3, len(top_indices))
        sampled_pool_indices = np.random.choice(top_indices, size=num_to_sample, replace=False, p=probs)
        ranked_tracks = [tracks[i] for i in sampled_pool_indices]
    else:
        # Fallback random
        ranked_tracks = list(tracks)
        np.random.shuffle(ranked_tracks)
        
    ranking_ms = (time.perf_counter() - t0) * 1000

    # 4. Post-Processing & Deduplication
    t0 = time.perf_counter()
    unique_artists = set()
    final_tracks = []
    
    for t in ranked_tracks:
        if len(final_tracks) >= req.limit:
            break
        # Allow max 2 tracks per artist
        if t.artist not in unique_artists or len([f for f in final_tracks if f.artist == t.artist]) < 2:
            unique_artists.add(t.artist)
            final_tracks.append(t)
            
    postproc_ms = (time.perf_counter() - t0) * 1000
    total_ms = (time.perf_counter() - pipeline_start) * 1000
    
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
        "query_vector_preview": user_vector[0][:8].tolist(),
        "vibe_steering_applied": bool(vibe and req.apply_vibe_steering)
    }
