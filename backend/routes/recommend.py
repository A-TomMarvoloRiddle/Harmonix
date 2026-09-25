from fastapi import APIRouter, Depends
from pydantic import BaseModel
import numpy as np
from typing import List
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

@router.post("/vibe")
async def update_vibe(vibe: VibeState):
    await set_vibe_studio_state(str(vibe.user_id), vibe.model_dump())
    return {"status": "updated"}

@router.post("/recommend")
async def get_recommendations(req: RecommendRequest, db: AsyncSession = Depends(get_db)):
    # 1. Fetch Vibe State
    vibe = await get_vibe_studio_state(str(req.user_id))
    
    # 2. Compute query vector (mocked logic for Vibe Steering)
    # Start with a random user vector (in production, fetched from Two-Tower User Tower)
    user_vector = np.random.randn(1, EMBEDDING_DIM).astype(np.float32)
    
    if req.apply_vibe_steering and vibe:
        energy = float(vibe.get("energy", 0.5))
        valence = float(vibe.get("valence", 0.5))
        # Simple projection heuristic for mock
        vibe_modifier = np.ones((1, EMBEDDING_DIM)) * (energy + valence - 1.0)
        query_vector = user_vector + 0.1 * vibe_modifier.astype(np.float32)
    else:
        query_vector = user_vector

    # Normalize
    query_vector = query_vector / np.linalg.norm(query_vector)

    # 3. Retrieve from FAISS
    distances, candidate_faiss_ids = search_candidates(query_vector, top_k=req.limit * 5)
    
    # 4. Mock Ranking & Deduplication
    unique_artists = set()
    final_tracks = []
    
    # Map faiss IDs back to DB IDs (assuming 1:1 for simplicity in Phase 1)
    db_ids = [int(i) for i in candidate_faiss_ids[0] if i != -1]
    
    if db_ids:
        # Fetch tracks
        result = await db.execute(select(Track).where(Track.id.in_(db_ids)))
        tracks = result.scalars().all()
        
        for t in tracks:
            if len(final_tracks) >= req.limit:
                break
            if t.artist not in unique_artists or len([f for f in final_tracks if f.artist == t.artist]) < 2:
                unique_artists.add(t.artist)
                final_tracks.append({
                    "id": t.id,
                    "title": t.title,
                    "artist": t.artist
                })

    return {"user_id": req.user_id, "tracks": final_tracks}
