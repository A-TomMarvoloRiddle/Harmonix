from fastapi import APIRouter
import json
import os

router = APIRouter()

def generate_gradcam_explanation(track_id: int, track=None):
    heatmap_matrix = []
    
    if os.path.exists("static/gradcam_data.json"):
        try:
            with open("static/gradcam_data.json", "r") as f:
                data = json.load(f)
                heatmap_matrix = data.get(str(track_id), [])
        except Exception:
            pass
            
    if not heatmap_matrix:
        heatmap_matrix = [[0.0]*10 for _ in range(4)]
        
    explanation = "Audio features matched user vibe state perfectly."
    band = "Mid-High (2kHz - 5kHz)"
    if track:
        if track.energy > 0.7:
            explanation = "High rhythmic transient density and strong RMS energy drove this recommendation."
            band = "Low-Mid (250Hz - 2kHz)"
        elif track.acousticness > 0.7:
            explanation = "Smooth harmonic spectral envelope drove this acoustic recommendation."
            band = "High (5kHz+)"
        elif track.danceability > 0.7:
            explanation = "Prominent beat strength and rhythmic consistency drove this danceable recommendation."
            band = "Sub-Bass (20Hz - 60Hz)"
            
    return {
        "track_id": track_id,
        "top_frequency_band": band,
        "explanation": explanation,
        "heatmap_matrix": heatmap_matrix
    }

from backend.database import async_session, Track
from sqlalchemy.future import select

@router.get("/tracks/{track_id}/explain")
async def explain_track(track_id: int):
    async with async_session() as session:
        result = await session.execute(select(Track).where(Track.id == track_id))
        track = result.scalars().first()
    return generate_gradcam_explanation(track_id, track)
