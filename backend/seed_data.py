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

def generate_synthetic_audio(duration=30, sr=22050, bpm=120, energy=0.5):
    """Generate synthetic audio if no files provided"""
    t = np.linspace(0, duration, int(sr * duration))
    # Base frequency
    freq = np.random.uniform(100, 800)
    y = np.sin(2 * np.pi * freq * t)
    # Add harmonics based on energy
    for i in range(2, 5):
        y += (energy / i) * np.sin(2 * np.pi * (freq * i) * t)
    
    # Add rhythm pulses based on bpm
    beat_interval = 60.0 / bpm
    pulse = np.zeros_like(y)
    for i in np.arange(0, duration, beat_interval):
        idx = int(i * sr)
        if idx < len(pulse):
            pulse[idx:idx+1000] = 1.0
            
    y = y * 0.5 + pulse * energy * 0.5
    return y.astype(np.float32), sr

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
    with torch.no_grad():
        embedding = model(mel_tensor)
        embedding_np = embedding.numpy()[0]
        
    # 3. Mock Grad-CAM (since we just ran forward pass without hooks for simplicity in seed)
    # We generate a heatmap based on the actual mel energy distribution
    heatmap = (mel_input - mel_input.min()) / (mel_input.max() - mel_input.min() + 1e-8)
    heatmap = np.clip(heatmap * 1.5, 0, 1) # Boost contrast
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='gray')
    plt.imshow(heatmap, aspect='auto', origin='lower', alpha=0.5, cmap='jet', extent=[0, mel_input.shape[1]*512/sr, 0, 8000])
    plt.tight_layout()
    gradcam_path = GRADCAM_DIR / f"track_{track_id}.png"
    plt.savefig(gradcam_path)
    plt.close()
    
    # Extract features
    duration_s = len(y) / sr
    bpm_tuple = librosa.beat.beat_track(y=y, sr=sr)
    bpm_val = bpm_tuple[0]
    if isinstance(bpm_val, np.ndarray):
        bpm_val = bpm_val.item() if bpm_val.size == 1 else np.mean(bpm_val)
    bpm = float(bpm_val)
    
    energy = min(1.0, float(np.mean(librosa.feature.rms(y=y))) * 10)
    
    # Assign genre based on simple heuristic
    genres = ["Electronic", "Rock", "Classical", "Hip-Hop", "Jazz"]
    genre = genres[int((bpm + energy * 100) % 5)]
    
    return {
        "embedding": embedding_np.tolist(),
        "duration_s": duration_s,
        "bpm": float(bpm),
        "energy": float(energy),
        "valence": float(np.random.uniform(0.2, 0.9)), # simplified
        "acousticness": float(1.0 - energy),
        "danceability": float(np.random.uniform(0.4, 0.9)),
        "genre": genre
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
        
        track_counter = 1
        
        # Process real files
        import re
        for file_path in audio_files:
            print(f"Processing: {file_path.name}")
            try:
                try:
                    y, sr = librosa.load(file_path, sr=22050, duration=30)
                except Exception as e1:
                    print(f"librosa native load failed: {e1}. Trying pydub...")
                    import os
                    ffmpeg_dir = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-essentials_build\bin")
                    if os.path.exists(ffmpeg_dir):
                        os.environ["PATH"] += os.pathsep + ffmpeg_dir
                        
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(file_path)[:30000] # first 30 seconds
                    audio = audio.set_frame_rate(22050).set_channels(1)
                    y = np.array(audio.get_array_of_samples(), dtype=np.float32)
                    y /= np.iinfo(audio.array_type).max
                    sr = 22050
                
                features = process_audio(y, sr, track_counter)
                
                stem = file_path.stem
                
                # New parse_filename logic inline
                stem = re.sub(r'\s*[\(\[][^\)\]]*[\)\]]', '', stem)
                stem = re.sub(r'\.mp3$', '', stem, flags=re.IGNORECASE)
                import unicodedata
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
                
                all_embeddings.append(features['embedding'])
                all_track_ids.append(track.id)
                track_counter += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
        await session.commit()
        
        # 3. Seed FAISS
        if all_embeddings:
            emb_matrix = np.array(all_embeddings, dtype=np.float32)
            add_embeddings(emb_matrix, all_track_ids)
            print(f"Added {len(all_track_ids)} tracks to FAISS index.")
            
        print("Database and FAISS seeded successfully!")

if __name__ == "__main__":
    asyncio.run(seed())
