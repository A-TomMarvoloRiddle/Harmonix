from fastapi import APIRouter
import numpy as np

router = APIRouter()

def generate_gradcam_explanation(track_id: int):
    # Mock implementation of Grad-CAM logic
    # In production, this hooks into the last conv layer of ResNet-18
    # and generates heatmaps over the mel-spectrogram
    
    mock_heatmap = np.random.rand(4, 10).tolist() # 4 frequency bands, 10 time slices
    
    return {
        "track_id": track_id,
        "top_frequency_band": "Mid-High (2kHz - 5kHz)",
        "explanation": "High rhythmic transient density in upper midrange drove valence recommendation",
        "heatmap_matrix": mock_heatmap
    }

@router.get("/tracks/{track_id}/explain")
async def explain_track(track_id: int):
    return generate_gradcam_explanation(track_id)
