import asyncio
import sys
import os
from pathlib import Path
import numpy as np
import librosa
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sqlalchemy.future import select
import json
import re
import unicodedata

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.database import engine, Base, async_session, User, Track
from backend.faiss_index import add_embeddings, EMBEDDING_DIM
from backend.ml.feature_extractor import AudioFeatureExtractor

AUDIO_DIR = Path("audio_samples")
SPECTROGRAM_DIR = Path("static/spectrograms")
GRADCAM_DIR = Path("static/gradcam")

# Ensure dirs exist
SPECTROGRAM_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

# Load model for feature extraction
model = AudioFeatureExtractor(embedding_dim=EMBEDDING_DIM)
model.eval()

def process_audio(y, sr, track_id):
    # 1. Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Save Spectrogram PNG
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='magma')
    plt.tight_layout()
    spec_path = SPECTROGRAM_DIR / f"track_{track_id}.png"
    plt.savefig(spec_path)
    plt.close()
    
    # Resize mel for model input (1, 1, 128, 128) - take middle 128 frames
    n_frames = mel_db.shape[1]
    if n_frames < 128:
        pad_width = 128 - n_frames
        mel_input = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        start = n_frames // 2 - 64
        mel_input = mel_db[:, start:start+128]
        
    mel_tensor = torch.tensor(mel_input).unsqueeze(0).unsqueeze(0).float()
    
    # 2. ResNet Embedding
    # Using torch.enable_grad() so Grad-CAM can compute backward pass
    with torch.enable_grad():
        mel_tensor.requires_grad_(True)
        # 3. Real Grad-CAM Heatmap
        heatmap = model.generate_gradcam(mel_tensor)
        
        # Forward pass for embedding (without gradients)
        with torch.no_grad():
            embedding = model(mel_tensor)
            embedding_np = embedding.numpy()[0]
    
    # Resize heatmap to match mel_input dimensions for display
    # Use PyTorch functional interpolation to avoid needing cv2
    h_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()
    heatmap_resized = torch.nn.functional.interpolate(
        h_tensor, size=(mel_input.shape[0], mel_input.shape[1]), mode='bilinear', align_corners=False
    ).squeeze().numpy()
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='gray')
    plt.imshow(heatmap_resized, aspect='auto', origin='lower', alpha=0.5, cmap='jet', extent=[0, mel_input.shape[1]*512/sr, 0, 8000])
    plt.tight_layout()
    gradcam_path = GRADCAM_DIR / f"track_{track_id}.png"
    plt.savefig(gradcam_path)
    plt.close()
    
    # Extract Real Acoustic Features (Zero Mocking)
    duration_s = len(y) / sr
    bpm_tuple = librosa.beat.beat_track(y=y, sr=sr)
    bpm_val = bpm_tuple[0]
    if isinstance(bpm_val, np.ndarray):
        bpm_val = bpm_val.item() if bpm_val.size == 1 else np.mean(bpm_val)
    bpm = float(bpm_val)
    
    # Energy: RMS
    energy = min(1.0, float(np.mean(librosa.feature.rms(y=y))) * 10)
    
    # Danceability: Variance of onset strength (beat regularity)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    danceability = float(np.clip(1.0 - (np.var(onset_env) / (np.max(onset_env) + 1e-8)), 0.0, 1.0))
    # Adjust for more realistic distribution
    danceability = min(1.0, danceability + 0.3)
    
    # Valence: Spectral Centroid ratio (Brighter = higher valence proxy)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    valence = float(np.clip(np.mean(cent) / 4000.0, 0.0, 1.0))
    
    acousticness = float(1.0 - energy)
    
    # Assign genre based on simple heuristic of audio features (since we don't have tags)
    genres = ["Electronic", "Rock", "Classical", "Hip-Hop", "Jazz"]
    genre_idx = int((bpm + energy * 100 + valence * 50) % 5)
    genre = genres[genre_idx]
    
    # Save the small heatmap matrix for the JSON explainability endpoint
    # Downsample to 4x10 grid for easy JSON transfer
    heatmap_json = torch.nn.functional.interpolate(
        h_tensor, size=(4, 10), mode='bilinear', align_corners=False
    ).squeeze().numpy().tolist()
    
    return {
        "embedding": embedding_np.tolist(),
        "duration_s": duration_s,
        "bpm": float(bpm),
        "energy": float(energy),
        "valence": float(valence),
        "acousticness": float(acousticness),
        "danceability": float(danceability),
        "genre": genre,
        "heatmap_json": heatmap_json
    }

async def seed():
    # 1. Clear and init DB
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    async with async_session() as session:
        user = User(username="admin_user", preferences={"genre": "electronic"})
        session.add(user)
        await session.commit()
        
        # 2. Process audio files
        audio_files = []
        if AUDIO_DIR.exists():
            for ext in ['.mp3', '.wav', '.m4a', '.mpeg', '.ogg', '.flac']:
                audio_files.extend(list(AUDIO_DIR.glob(f"*{ext}")))
                
        print(f"Found {len(audio_files)} audio files.")
        
        all_embeddings = []
        all_track_ids = []
        heatmap_data = {}
        
        track_counter = 1
        
        for file_path in audio_files:
            print(f"Processing: {file_path.name}")
            try:
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        y, sr = librosa.load(file_path, sr=22050, duration=30)
                except Exception as e1:
                    print(f"librosa native load failed: {e1}. Trying pydub w/ imageio-ffmpeg...")
                    import imageio_ffmpeg
                    from pydub import AudioSegment
                    
                    # Dynamically set ffmpeg without requiring system PATH
                    AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
                    
                    audio = AudioSegment.from_file(file_path)[:30000] # first 30 seconds
                    audio = audio.set_frame_rate(22050).set_channels(1)
                    y = np.array(audio.get_array_of_samples(), dtype=np.float32)
                    y /= np.iinfo(audio.array_type).max
                    sr = 22050
                
                features = process_audio(y, sr, track_counter)
                
                stem = file_path.stem
                stem = re.sub(r'\s*[\(\[][^\)\]]*[\)\]]', '', stem)
                stem = re.sub(r'\.mp3$', '', stem, flags=re.IGNORECASE)
                stem = unicodedata.normalize('NFKC', stem).strip()
                
                if '_-_' in stem:
                    parts = stem.split('_-_', 1)
                    artist = parts[0].replace('_', ' ').strip().title()
                    title = parts[1].replace('_', ' ').strip().title()
                elif re.search(r'\s+-\s+', stem):
                    parts = re.split(r'\s+-\s+', stem, 1)
                    artist = parts[0].strip().title()
                    title = parts[1].strip().title()
                elif ' – ' in stem:
                    parts = stem.split(' – ', 1)
                    artist = parts[0].strip().title()
                    title = parts[1].strip().title()
                else:
                    artist = "Various Artists"
                    title = re.sub(r'^\d+[\s_]+', '', stem.replace('_', ' ')).strip().title()
                
                track = Track(
                    title=title,
                    artist=artist,
                    genre=features['genre'],
                    duration_s=features['duration_s'],
                    bpm=features['bpm'],
                    energy=features['energy'],
                    valence=features['valence'],
                    acousticness=features['acousticness'],
                    danceability=features['danceability'],
                    file_path=str(file_path),
                    embedding=features['embedding']
                )
                session.add(track)
                await session.flush()
                
                # Store heatmap for explainability endpoint
                heatmap_data[str(track.id)] = features['heatmap_json']
                
                all_embeddings.append(features['embedding'])
                all_track_ids.append(track.id)
                track_counter += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()
                
        await session.commit()
        
        # Save heatmaps to a JSON file for the explainability router to read
        with open("static/gradcam_data.json", "w") as f:
            json.dump(heatmap_data, f)
        
        # 3. Seed FAISS
        if all_embeddings:
            emb_matrix = np.array(all_embeddings, dtype=np.float32)
            add_embeddings(emb_matrix, all_track_ids)
            print(f"Added {len(all_track_ids)} tracks to FAISS index.")
            
        print("Database and FAISS seeded successfully!")

if __name__ == "__main__":
    asyncio.run(seed())
