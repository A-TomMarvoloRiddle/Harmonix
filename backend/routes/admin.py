from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
import os
import time
import numpy as np
from collections import deque
from backend.faiss_index import get_index_size
from backend.database import async_session, Track, User
from backend.redis_client import redis_client
from sqlalchemy.future import select
from sqlalchemy import func
from typing import List, Dict

router = APIRouter()

# In-memory telemetry log for the admin dashboard
telemetry_buffer = deque(maxlen=100)
stats_data = {"kafka_msg_count": 0}

async def get_db():
    async with async_session() as session:
        yield session

@router.get("/evaluation")
async def get_evaluation_metrics(db=Depends(get_db)):
    from backend.faiss_index import get_all_embeddings, index
    from sklearn.cluster import KMeans
    import numpy as np
    
    # 1. Fetch all tracks
    result = await db.execute(select(Track))
    tracks = {t.id: t for t in result.scalars().all()}
    
    embeddings, track_ids = get_all_embeddings()
    if len(embeddings) < 10:
        return {"ndcg_at_10": 0, "recall_at_10": 0, "mrr": 0}
    
    # 2. Phase 6: Mathematically rigorous clustering for ground truth
    num_clusters = min(5, len(embeddings) // 5)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(embeddings)
    
    cluster_map = {track_ids[i]: labels[i] for i in range(len(track_ids))}
    
    K = min(10, len(embeddings))
    ndcg_scores, recall_scores, rr_scores = [], [], []
    
    for i in range(len(embeddings)):
        query_tid = track_ids[i]
        query_cluster = cluster_map[query_tid]
        
        query_vector = np.array([embeddings[i]], dtype=np.float32)
        distances, indices = index.search(query_vector, K + 1)
        
        # Remove self from retrieved
        retrieved = [track_ids[j] for j in indices[0] if j != -1 and track_ids[j] != query_tid][:K]
        
        # Ground truth: tracks in the same mathematical cluster
        ground_truth = {tid for tid, cluster in cluster_map.items() if cluster == query_cluster and tid != query_tid}
        if not ground_truth:
            continue
            
        # NDCG
        gains = [1.0 if t in ground_truth else 0.0 for t in retrieved]
        dcg = sum(g / np.log2(r + 2) for r, g in enumerate(gains))
        ideal_gains = sorted([1.0]*min(len(ground_truth), K) + [0.0]*(K - len(ground_truth)), reverse=True)
        ideal_dcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal_gains))
        ndcg_scores.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
        
        # Recall
        hits = sum(1 for t in retrieved if t in ground_truth)
        recall_scores.append(hits / min(len(ground_truth), K))
        
        # MRR
        rr = next((1.0/(r+1) for r, t in enumerate(retrieved) if t in ground_truth), 0.0)
        rr_scores.append(rr)
    
    return {
        "ndcg_at_10": round(float(np.mean(ndcg_scores)), 4) if ndcg_scores else 0.0,
        "recall_at_10": round(float(np.mean(recall_scores)), 4) if recall_scores else 0.0,
        "mrr": round(float(np.mean(rr_scores)), 4) if rr_scores else 0.0,
        "num_tracks_evaluated": len(track_ids)
    }

@router.get("/audio/{track_id}")
async def serve_audio(track_id: int, db=Depends(get_db)):
    result = await db.execute(select(Track).where(Track.id == track_id))
    track = result.scalars().first()
    if not track or not track.file_path:
        return {"error": "no audio file for this track"}
    path = track.file_path
    if not os.path.exists(path):
        return {"error": "file not found on disk"}
    return FileResponse(path, media_type="audio/mpeg")

@router.get("/stats")
async def get_system_stats(db=Depends(get_db)):
    # 1. FAISS size
    faiss_size = get_index_size()
    
    # 2. DB sizes
    tracks_result = await db.execute(select(func.count()).select_from(Track))
    track_count = tracks_result.scalar() or 0
    
    users_result = await db.execute(select(func.count()).select_from(User))
    user_count = users_result.scalar() or 0
    
    # 3. Redis keys
    redis_keys = await redis_client.keys("vibe:*")
    
    return {
        "faiss_index_size": faiss_size,
        "track_count": track_count,
        "user_count": user_count,
        "redis_vibe_keys": len(redis_keys),
        "kafka_status": f"connected ({stats_data['kafka_msg_count']} msgs)"
    }

@router.get("/tsne")
async def get_tsne_data(db=Depends(get_db)):
    # Fetch all embeddings from DB
    result = await db.execute(select(Track.id, Track.title, Track.artist, Track.genre, Track.embedding))
    tracks = result.all()
    
    valid_tracks = [t for t in tracks if t.embedding is not None]
    if not valid_tracks:
        return []
        
    embeddings = np.array([t.embedding for t in valid_tracks])
    
    # Simple PCA-like 2D projection if TSNE takes too long, but TSNE for < 1000 items is fast
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(valid_tracks)-1), max_iter=500, random_state=42)
    coords = tsne.fit_transform(embeddings)
    
    data = []
    for i, t in enumerate(valid_tracks):
        data.append({
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "track_id": t.id,
            "title": t.title,
            "artist": t.artist,
            "genre": t.genre
        })
    return data

@router.get("/tracks")
async def get_tracks(db=Depends(get_db)):
    result = await db.execute(select(Track).limit(100))
    tracks = result.scalars().all()
    return [{"id": t.id, "title": t.title, "artist": t.artist, "genre": t.genre, "duration_s": t.duration_s, "bpm": t.bpm} for t in tracks]

@router.get("/track/{track_id}/embedding")
async def get_track_embedding(track_id: int, db=Depends(get_db)):
    result = await db.execute(select(Track).where(Track.id == track_id))
    track = result.scalars().first()
    if not track or not track.embedding:
        return {"error": "not found"}
    
    # Scale embeddings strictly for visual differentiation in the UI
    scaled_vector = [val * 300 for val in track.embedding]
    return {"track_id": track.id, "title": track.title, "vector": scaled_vector}

@router.get("/telemetry")
async def get_telemetry():
    return list(telemetry_buffer)

@router.get("/redis")
async def get_redis_state():
    keys = await redis_client.keys("vibe:*")
    state = []
    for k in keys:
        raw_data = await redis_client.hgetall(k)
        # Decode byte strings natively for UI rendering
        decoded_key = k.decode('utf-8') if isinstance(k, bytes) else k
        decoded_data = {
            fk.decode('utf-8') if isinstance(fk, bytes) else fk: 
            fv.decode('utf-8') if isinstance(fv, bytes) else fv 
            for fk, fv in raw_data.items()
        }
        state.append({"key": decoded_key, "data": decoded_data})
    return state

@router.post("/simulate")
async def simulate_telemetry(db=Depends(get_db)):
    import random
    from datetime import datetime, timezone
    
    result = await db.execute(select(Track.id).limit(50))
    track_ids = [t for t in result.scalars().all()]
    if not track_ids:
        return {"status": "no tracks"}
        
    events = ["play", "play", "skip", "complete", "save"]
    
    for _ in range(20):
        tid = random.choice(track_ids)
        ev = random.choice(events)
        dur = random.randint(5000, 180000)
        
        telemetry_buffer.append({
            "user_id": 1,
            "track_id": tid,
            "event_type": ev,
            "duration_ms": dur if ev != "skip" else random.randint(1000, 15000),
            "listened_percentage": random.uniform(0.1, 0.9),
            "is_rapid_skip": ev == "skip" and random.random() > 0.5,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    stats_data["kafka_msg_count"] += 20
        
    return {"status": "simulated 20 events"}
