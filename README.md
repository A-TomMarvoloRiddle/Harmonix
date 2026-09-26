<div align="center">

# 🎵 Harmonix
### A Deep Learning-Based Music Recommendation & Ranking Engine

*"MLOps-enabled Multimodal Recommender Systems, utilizing Deep Learning and Vector Search within a Highly Scalable, Low-Latency Microservices Architecture to deliver real-time personalized content."*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue?style=flat-square)](https://github.com/facebookresearch/faiss)
[![Redis](https://img.shields.io/badge/Redis-In--Memory_Cache-DC382D?style=flat-square&logo=redis&logoColor=white)](https://redis.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Async-336791?style=flat-square&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Kafka](https://img.shields.io/badge/Apache_Kafka-Event_Streaming-231F20?style=flat-square&logo=apache-kafka&logoColor=white)](https://kafka.apache.org)

</div>

---

## Table of Contents

1. [Abstract](#abstract)
2. [Project Domain & Significance](#project-domain--significance)
3. [System Architecture](#system-architecture)
4. [The Two-Stage Recommendation Funnel](#the-two-stage-recommendation-funnel)
5. [Full ML Pipeline — From Audio to Recommendation](#full-ml-pipeline--from-audio-to-recommendation)
6. [Data Flow Diagram](#data-flow-diagram)
7. [Use Case Diagram](#use-case-diagram)
8. [Sequence Diagram — Vibe Steering Loop](#sequence-diagram--vibe-steering-loop)
9. [Technology Stack](#technology-stack)
10. [Database Schema & ER Design](#database-schema--er-design)
11. [API Reference](#api-reference)
12. [Frontend Dashboard — Module Breakdown](#frontend-dashboard--module-breakdown)
13. [Recommendation Quality Metrics](#recommendation-quality-metrics)
14. [Evaluation & Benchmarks](#evaluation--benchmarks)
15. [Algorithms & Research Foundation](#algorithms--research-foundation)
16. [Project Structure](#project-structure)
17. [Setup & Installation](#setup--installation)
18. [Running the Application](#running-the-application)
19. [Seeding the Database](#seeding-the-database)
20. [Testing](#testing)
21. [Work Distribution](#work-distribution)
22. [Future Roadmap](#future-roadmap)
23. [References](#references)

---

## Abstract

Harmonix is a production-grade, end-to-end music recommendation engine that operates at the intersection of **Deep Learning**, **Vector Search**, **Real-Time Stream Processing**, and **MLOps**. It implements an industry-standard multi-stage funnel architecture—mirroring the internal systems of platforms like Spotify, YouTube Music, and Gaana—to deliver personalized music recommendations in under 200 milliseconds.

The system continuously ingests implicit user behavioral signals (playback duration, skip rates, track completion percentages, crossfade transitions) via Apache Kafka and uses them to dynamically reshape a user's **Vibe State**—a 4-dimensional audio-feature vector (energy, valence, acousticness, danceability) cached in Redis. This vector is used to steer the FAISS approximate nearest-neighbor search at query time, ensuring that recommendations are not static but adapt instantaneously to the user's current listening context.

Audio feature extraction is performed offline using **Librosa** (BPM, energy, spectral centroid, rhythm), **pydub** and **FFMPEG** for format normalization, and a **ResNet-18 CNN** forward pass on mel-spectrograms to produce 128-dimensional dense audio embeddings. **Grad-CAM heatmaps** are generated over these spectrograms to provide transparent, visual explainability for every recommendation.

---

## Project Domain & Significance

Harmonix sits at the intersection of five high-impact engineering domains:

| Domain | Keywords |
|---|---|
| **AI & Deep Learning** | Two-Tower Networks, ResNet-18, Mel-Spectrogram CNNs, Grad-CAM, Representation Learning |
| **Information Retrieval** | Vector Search, FAISS ANN, Semantic Search, Dense Retrieval, Embedding Indexing |
| **MLOps** | CI/CD, Continuous Inference, Model Registry, Drift Monitoring, A/B Testing |
| **Distributed Systems** | Sub-200ms Latency, Microservices, Redis Caching, Kafka Event Streaming |
| **Data Engineering** | Real-Time Stream Processing, Feature Stores, Telemetry Pipelines, Batch Audio Ingestion |

### The "Choice Overload" Problem
Modern music platforms expose users to tens of millions of tracks. Static recommendation systems that ignore real-time context (time of day, current mood, listening fatigue) fail to keep users engaged. Harmonix solves this by treating each listening event as a signal to continuously update the recommendation trajectory—not just at session start, but after every skip, save, and crossfade transition.

---

## System Architecture

```
                          ┌─────────────────────────────────────────────────────────┐
                          │              BROWSER (localhost:8000)                    │
                          │                                                          │
                          │  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐  │
                          │  │🎵 Player   │  │🎛️ Vibe Studio│  │ 🔬 ML Lab /     │  │
                          │  │ Shell      │  │ (4 Sliders + │  │ 📊 Data & Metrics│  │
                          │  │ Dual-Audio │  │  Radar Chart)│  │ (Admin Views)   │  │
                          │  └─────┬──────┘  └──────┬───────┘  └────────┬────────┘  │
                          │        │                 │                   │            │
                          └────────┼─────────────────┼───────────────────┼────────────┘
                                   │ REST/JSON        │ Vector Overrides  │ Admin Queries
                                   ▼                  ▼                   ▼
                    ┌──────────────────────────────────────────────────────────────────┐
                    │              FastAPI API Gateway  (Uvicorn ASGI)                │
                    │                                                                  │
                    │  /api/v1/recommend   /api/v1/telemetry   /api/v1/vibe           │
                    │  /api/v1/admin/*     /api/v1/tracks/*/explain                   │
                    │                                                                  │
                    │   ┌─────────────────────────────────────────────────────────┐   │
                    │   │           Business Logic & Post-Processing Engine        │   │
                    │   │   • Artist Deduplication  • Novelty Injection           │   │
                    │   │   • Intra-List Diversity  • Crossfade Telemetry         │   │
                    │   └───────────────────────────────────────────────────────┬─┘   │
                    └─────────────────────────────────────────────────────────── ┼────┘
                                                                                 │
             ┌───────────────────────────────── ONLINE SERVING ──────────────────┼────────────────────────────┐
             │                                                                    │                            │
             │  ┌─────────────────────┐                         ┌────────────────▼────────────────────────┐  │
             │  │   Apache Kafka      │                         │    Stage 3: Precision Ranking            │  │
             │  │  (Event Streaming)  │◄── Telemetry Events     │    Deep & Wide Network / MMoE            │  │
             │  │  • play, pause      │                         │    • Re-ranks 500 candidates             │  │
             │  │  • skip, heartbeat  │                         │    • Scores by: time of day, context     │  │
             │  │  • crossfade_start  │                         │    • Assigns engagement probability      │  │
             │  └──────────┬──────────┘                         └────────────────▲────────────────────────┘  │
             │             │                                                      │ (~500 Candidates)          │
             │  ┌──────────▼──────────┐                         ┌────────────────┴────────────────────────┐  │
             │  │   Redis Cluster     │ ──► Vibe State Vector ──►│    Stage 2: Candidate Retrieval         │  │
             │  │  (Session Cache)    │                         │    FAISS ANN Index (IndexFlatIP)         │  │
             │  │  • Vibe States      │                         │    • Query vector = random + vibe shift  │  │
             │  │  • Alpha Blending   │                         │    • top_K = limit × 5                   │  │
             │  └─────────────────────┘                         └────────────────▲────────────────────────┘  │
             └────────────────────────────────────────────────────────────────── ┼──────────────────────────┘
                                                                                  │
             ┌────────────────────────────────── OFFLINE PIPELINE ────────────── ┼──────────────────────────┐
             │                                                                    │                           │
             │  ┌─────────────────────┐                         ┌────────────────┴────────────────────────┐ │
             │  │   PostgreSQL        │                         │    Stage 1: Dense Embedding Space        │ │
             │  │  (Persistent Store) │──► Track Metadata ─────►│    128-dimensional float32 vectors       │ │
             │  │  • Users            │                         │    (FAISS IndexFlatIP, L2 normalized)    │ │
             │  │  • Tracks           │                         └────────────────▲────────────────────────┘ │
             │  │  • Playlists        │                                          │                           │
             │  │  • ListeningEvents  │           ┌─────────────────────────────┘                           │
             │  └─────────────────────┘           │                                                         │
             │                                    │ PyTorch ResNet-18 Forward Pass (128-d)                  │
             │  ┌─────────────────────────────────┴────────────────────────────────────┐                   │
             │  │              Audio Preprocessing Pipeline                             │                   │
             │  │   audio_samples/  ──► FFMPEG/pydub ──► Librosa ──► Mel-Spectrogram  │                   │
             │  │   *.mp3, *.wav,         (decode)       (feature    (128×128 tensor)  │                   │
             │  │   *.m4a, *.mpeg                         extract)                     │                   │
             │  │                                                   ──► ResNet-18       │                   │
             │  │                                                   ──► Grad-CAM PNG   │                   │
             │  │                                                   ──► Spectrogram PNG │                   │
             │  └──────────────────────────────────────────────────────────────────────┘                   │
             └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## The Two-Stage Recommendation Funnel

The core architecture separates the recommendation problem into two distinct computational stages, a pattern proven at industrial scale by YouTube, Spotify, and Pinterest.

### Stage 1 — Candidate Generation (Retrieval)

**Goal:** Reduce millions of tracks to ~500 high-recall candidates in under 20ms.

The user's **Vibe State** (a 4-dimensional vector of energy, valence, acousticness, danceability stored in Redis) is used to construct a **biased query vector** in 128-dimensional embedding space. This vector is not purely random—it is shifted in the direction indicated by the user's current vibe:

```python
user_vector = np.random.randn(1, EMBEDDING_DIM).astype(np.float32)  # base
vibe_modifier = np.zeros((1, EMBEDDING_DIM))
vibe_modifier[0, 0:32]   = energy - 0.5       # energy dimension cluster
vibe_modifier[0, 32:64]  = valence - 0.5      # valence dimension cluster
vibe_modifier[0, 64:96]  = danceability - 0.5 # danceability cluster
vibe_modifier[0, 96:128] = 0.5 - acousticness # acousticness (inverse)

query_vector = normalize(user_vector + 0.5 * vibe_modifier)
```

This query is then used to search the FAISS `IndexFlatIP` (inner product / cosine similarity) across all indexed track embeddings, returning `top_k = limit × 5` nearest neighbors.

### Stage 2 — Precision Ranking (Scoring)

**Goal:** Re-rank the 500 candidates using contextual signals.

The retrieved candidates are passed through a ranking layer that assigns engagement probability scores, shuffled by adding controlled random noise to simulate the Deep & Wide scoring mechanism. Final artist-level deduplication (max 2 tracks per artist) ensures diversity before the final playlist is returned.

### Stage 3 — Post-Processing & Business Logic

- **Artist Deduplication:** Max 2 tracks per artist in any recommendation batch.
- **Intra-List Diversity (ILD):** Average pairwise cosine distance across recommended track embeddings.
- **Novelty Score:** Inverse-popularity heuristic (`track_id % 10 / 10.0`) surfacing long-tail content.
- **Genre Spread:** Count of distinct genres in the final list.

---

## Full ML Pipeline — From Audio to Recommendation

```
audio_samples/*.{mp3,wav,m4a,mpeg}
        │
        ▼  FFMPEG/pydub → normalize codec, resample to 22050 Hz mono
   Decoded Waveform (y, sr)
        │
        ├─► librosa.feature.melspectrogram(n_mels=128)
        │   → Mel Spectrogram (128 × T frames)
        │
        ├─► Feature Extraction (no API needed):
        │   • duration_s  = len(y) / sr
        │   • bpm         = librosa.beat.beat_track(y, sr)[0]
        │   • energy      = normalize(mean(librosa.feature.rms(y=y)))
        │   • danceability= derived from beat strength variance
        │   • valence     = derived from spectral centroid (brighter = more positive)
        │   • acousticness= inverse of high-frequency content ratio
        │   • genre       = cluster from BPM + energy (5 classes: Electronic, Rock,
        │                   Classical, Hip-Hop, Jazz)
        │
        ├─► matplotlib.imshow() → static/spectrograms/track_{id}.png
        │
        ▼  resize Mel Spectrogram to (1, 1, 128, 128) tensor
   ResNet-18 AudioFeatureExtractor.forward()
        │
        ├─► AdaptiveAvgPool2d → FC(512, 128) → L2 Normalize
        │   → 128-d dense embedding stored in:
        │     • PostgreSQL: Track.embedding (JSON column)
        │     • FAISS IndexFlatIP: for ANN search
        │
        ▼  Grad-CAM on final convolutional layer
   Activation heatmap overlay on Mel Spectrogram
        │
        └─► matplotlib overlay → static/gradcam/track_{id}.png
            (used in "Why This Track?" explainability drawer)
```

### Filename → Metadata Parsing

Track titles and artists are extracted intelligently from filenames without any external music API:

```python
# Regex-based parsing handles formats like:
# "Amit_Trivedi_-_Iktara_(mp3.pm).mp3.mpeg"  → Title: "Iktara", Artist: "Amit Trivedi"
# "Sia_-_Cheap_trills_(mp3.pm).mp3.mpeg"      → Title: "Cheap Trills", Artist: "Sia"
# "1_Farmhouse_Alien.m4a"                      → Title: "Farmhouse Alien", Artist: "Various Artists"

def parse_filename(stem: str) -> tuple[str, str]:
    stem = re.sub(r'\s*[\(\[][^\)\]]*[\)\]]', '', stem)  # strip junk suffixes
    stem = re.sub(r'\.mp3$', '', stem, flags=re.IGNORECASE)
    stem = unicodedata.normalize('NFKC', stem).strip()
    
    if '_-_' in stem:
        artist, title = stem.split('_-_', 1)
    elif re.search(r'\s+-\s+', stem):
        artist, title = re.split(r'\s+-\s+', stem, 1)
    else:
        artist, title = "Various Artists", re.sub(r'^\d+[\s_]+', '', stem)
    
    return title.replace('_', ' ').strip().title(), artist.replace('_', ' ').strip().title()
```

---

## Data Flow Diagram

```
                    ┌─────────────────────────────────────────────┐
                    │             User Interaction                 │
                    │  (Play, Pause, Skip, Save, Crossfade, Vibe) │
                    └───────────────────┬─────────────────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │      Telemetry Emitter       │
                         │  sendTelemetry(event, ctx)   │
                         │  • event_type                │
                         │  • duration_ms               │
                         │  • listened_percentage       │
                         │  • is_rapid_skip (<8%)       │
                         │  • context: {bpm, genre,     │
                         │    energy, valence}          │
                         └──────────────┬───────────────┘
                                        │ POST /api/v1/telemetry
                         ┌──────────────▼───────────────┐
                         │     Telemetry Processor       │
                         │     (FastAPI Route)           │
                         └──┬────────────┬──────────────┘
                            │            │
               ┌────────────▼──┐    ┌────▼────────────────────┐
               │ PostgreSQL    │    │ Redis Vibe State Update  │
               │ ListeningEvent│    │ α-blend if pct > 30%:   │
               │ INSERT        │    │ vibe[energy] = 0.95 *   │
               └───────────────┘    │ old + 0.05 * track.eng  │
                                    └────────────┬────────────┘
                                                 │
               ┌──────────────────────────────────▼───────────────────────────────────┐
               │                POST /api/v1/recommend                                 │
               │                                                                       │
               │  1. Redis FETCH vibe state         (~1ms)                            │
               │  2. Construct biased query vector  (<1ms)                            │
               │  3. FAISS ANN search (limit × 5)   (~3ms)                            │
               │  4. PostgreSQL hydrate metadata    (~10ms)                            │
               │  5. Mock D&W ranking scores        (~2ms)                            │
               │  6. Artist dedup + diversity calc  (~1ms)                            │
               │                                              Total: <20ms             │
               └─────────────────────────────┬─────────────────────────────────────────┘
                                             │
                         ┌───────────────────▼────────────────────┐
                         │         Response Payload               │
                         │  • tracks[]   (ranked playlist)        │
                         │  • pipeline_metrics (per-stage ms)     │
                         │  • quality_metrics (ILD, novelty, etc) │
                         │  • query_vector_preview (first 8 dims) │
                         │  • vibe_steering_applied: bool         │
                         └────────────────────────────────────────┘
```

---

## Use Case Diagram

```
                        ┌─────────────────────────────────────────────┐
                        │              Harmonix System                 │
                        │                                              │
  ┌──────┐              │  ┌────────────────────────┐                 │
  │      │◄─ Play/Pause ──►│ UC1: Audio Playback     │                 │
  │      │◄─ Crossfade  ──►│     (Dual-Audio Engine) │                 │
  │ User │                 └────────────┬───────────┘                 │
  │      │◄─ Skip/Next  ──►┌────────────▼───────────┐                 │
  │      │◄─ Save Track ──►│ UC2: Implicit Telemetry │                 │
  └──┬───┘                 │     Emission            │                 │
     │                     └────────────┬───────────┘                 │
     │                                  │ (includes)                   │
     │                     ┌────────────▼───────────┐                 │
     ├─ Vibe Studio ──────►│ UC3: Vector-Steered     │                 │
     │   (Sliders)         │     Recommendations     │                 │
     │                     └────────────────────────┘                 │
     │                     ┌────────────────────────┐                 │
     └─ View Radar ───────►│ UC4: Vibe State Monitor │                 │
         Gauges            │     (Radar + Gauges)    │                 │
                           └────────────────────────┘                 │
                                                                        │
  ┌───────┐               ┌────────────────────────┐                  │
  │ Admin │◄─ Embed Inspect►│ UC5: FAISS Embedding   │                  │
  │  /    │               │     Inspector (128-D)   │                  │
  │  ML   │◄─ Eval Metrics►│ UC6: Offline Evaluation│                  │
  │ Engr  │               │     (NDCG, MRR, Recall) │                  │
  │       │◄─ Latency View►│ UC7: Pipeline Latency   │                  │
  └───────┘               │     Waterfall           │                  │
                          └────────────────────────┘                  │
                          ┌────────────────────────┐                  │
                          │ UC8: Simulate 20 Events│                  │
                          │     (Telemetry Storm)  │                  │
                          └────────────────────────┘                  │
                          ┌────────────────────────┐                  │
                          │ UC9: t-SNE Vector Space│                  │
                          │     Visualization      │                  │
                          └────────────────────────┘                  │
                        └─────────────────────────────────────────────┘
```

---

## Sequence Diagram — Vibe Steering Loop

```
User       Audio Player (JS)     FastAPI Backend      Redis         FAISS        PostgreSQL
 │                │                     │                │               │               │
 │──Play Track───►│                     │                │               │               │
 │                │──POST /telemetry ──►│                │               │               │
 │                │   {event:"play",    │──FETCH vibe:1─►│               │               │
 │                │    context:{bpm,    │◄──{energy:0.6}─┤               │               │
 │                │    genre,energy}}   │                │               │               │
 │                │◄────200 OK──────────│                │               │               │
 │                │                     │                │               │               │
 │ [30s heartbeat]│                     │                │               │               │
 │                │──POST /telemetry ──►│                │               │               │
 │                │   {event:"heartbeat"│                │               │               │
 │                │    pct: 0.45}       │                │               │               │
 │                │                     │──α-blend ─────►│               │               │
 │                │                     │  vibe[energy]= │               │               │
 │                │                     │  0.95*0.6 +    │               │               │
 │                │                     │  0.05*track.e  │               │               │
 │                │◄────200 OK──────────│                │               │               │
 │                │                     │                │               │               │
 │ [Track ends /  │                     │                │               │               │
 │  Crossfade ]   │                     │                │               │               │
 │                │──POST /recommend ──►│                │               │               │
 │                │   {user_id:1,       │──FETCH vibe:1─►│               │               │
 │                │    limit:6,         │◄──{energy:0.62}┤               │               │
 │                │    apply_vibe:true} │                │               │               │
 │                │                     │──Build query   │               │               │
 │                │                     │  vector w/vibe │               │               │
 │                │                     │  shift ────────┼──ANN search──►│               │
 │                │                     │                │◄──top 30 IDs──┤               │
 │                │                     │────────────────┼───────────────┼──SELECT WHERE │
 │                │                     │                │               │   id IN [...]─►│
 │                │                     │                │               │◄──Track objs──┤
 │                │                     │──Artist dedup  │               │               │
 │                │                     │──Diversity calc│               │               │
 │                │◄──{tracks:[6],       │               │               │               │
 │                │   pipeline_metrics, │                │               │               │
 │                │   quality_metrics}──│                │               │               │
 │                │                     │                │               │               │
 │◄──New playlist │                     │                │               │               │
 │   auto-plays   │                     │                │               │               │
```

---

## Technology Stack

### Full Stack Overview

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Vanilla JS, Tailwind CSS (CDN), Chart.js (CDN) | Single-page dashboard, zero build step |
| **Audio Engine** | Dual HTML5 `Audio` elements (Web Audio API) | Seamless crossfading, granular telemetry |
| **API Gateway** | FastAPI (Python) + Uvicorn ASGI | Async request routing, OpenAPI docs |
| **ML Inference** | PyTorch 2.0, TorchVision (ResNet-18) | Audio embedding generation, Grad-CAM |
| **Vector Search** | FAISS (`IndexFlatIP`, CPU) | Sub-20ms ANN retrieval across all tracks |
| **Feature Extraction** | Librosa, pydub, FFMPEG | BPM, energy, valence, mel-spectrograms |
| **Session Cache** | Redis (`hgetall`/`hset`, hash maps) | Vibe state per user, alpha-blended in real time |
| **Event Streaming** | Apache Kafka (with in-memory fallback) | Telemetry event ingestion pipeline |
| **Relational DB** | PostgreSQL + SQLAlchemy asyncpg | Track metadata, user profiles, listening events |
| **Dimensionality Reduction** | scikit-learn t-SNE | 2D projection of 128-d embeddings for visualization |
| **Visualization** | Chart.js (Radar, Doughnut, Bar, Scatter) | Latency waterfalls, gauges, embedding inspector |
| **Explainability** | Grad-CAM (PyTorch hooks on ResNet-18 final conv layer) | Frequency-band heatmaps for each recommendation |

### Python Dependencies

```
fastapi
uvicorn
sqlalchemy[asyncio]
asyncpg
redis
faiss-cpu
numpy
torch
torchvision
torchaudio
librosa
pydub
audioread
ffmpeg-python
matplotlib
kafka-python-ng
scikit-learn
pytest
pytest-asyncio
httpx
```

---

## Database Schema & ER Design

### Relational Schema (PostgreSQL)

```
┌──────────────┐                 ┌────────────────────┐
│    users     │  1           ∞  │  listening_events  │
├──────────────┤─────────────────├────────────────────┤
│ id       PK  │                 │ id          PK     │
│ username  UK │                 │ user_id     FK     │
│ preferences  │                 │ track_id    FK     │
│  (JSON)      │                 │ duration_ms        │
└──────┬───────┘                 │ timestamp          │
       │ 1                       └─────────┬──────────┘
       │                                   │ ∞
       │ ∞                                 │
┌──────┴───────┐                           │ 1
│  playlists   │                  ┌────────┴───────────────────┐
├──────────────┤                  │          tracks             │
│ id       PK  │                  ├─────────────────────────────┤
│ user_id  FK  │                  │ id           PK             │
│ name         │                  │ title                       │
└──────────────┘                  │ artist                      │
                                  │ genre                       │
                                  │ duration_s                  │
                                  │ bpm          (Librosa)      │
                                  │ energy       (Librosa RMS)  │
                                  │ valence      (spectral ctr) │
                                  │ acousticness (HF ratio inv) │
                                  │ danceability (beat variance)│
                                  │ file_path                   │
                                  │ embedding    (JSON: 128-d)  │
                                  └─────────────────────────────┘
```

### Redis Schema — Vibe State

Each user's vibe state is stored as a Redis hash keyed by `vibe:{user_id}`:

```
HSET vibe:1 energy 0.72 valence 0.61 acousticness 0.38 danceability 0.85
```

Updated in real time using an alpha blend whenever a user plays more than 30% of a track, saves a track, or explicitly adjusts sliders in Vibe Studio:

```python
alpha = 0.20 if event_type == "save" else 0.05
new_value = float(old_value) * (1 - alpha) + track.feature * alpha
```

### FAISS Index Schema

- **Index Type:** `IndexFlatIP` (exact inner product, equivalent to cosine similarity on L2-normalized vectors)
- **Dimensionality:** 128-d float32
- **Mapping:** `_track_id_map: list[int]` — maps FAISS internal position index → PostgreSQL `track.id`

```python
# On startup, warm_up_faiss() loads all embeddings from PostgreSQL:
for track in all_tracks:
    embedding = np.array(track.embedding, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(embedding)
    index.add(embedding)
    _track_id_map.append(track.id)
```

---

## API Reference

All endpoints are available at `http://localhost:8000`. Full interactive docs at `/docs`.

### Recommendation & Vibe

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/recommend` | Get vibe-steered track recommendations |
| `POST` | `/api/v1/vibe` | Update user's Vibe State in Redis |

**POST /api/v1/recommend — Request:**
```json
{
  "user_id": 1,
  "limit": 6,
  "apply_vibe_steering": true
}
```

**POST /api/v1/recommend — Response:**
```json
{
  "user_id": 1,
  "tracks": [
    {
      "id": 7,
      "title": "Cheap Thrills",
      "artist": "Sia",
      "genre": "Electronic",
      "energy": 0.81,
      "valence": 0.74,
      "bpm": 93.4
    }
  ],
  "pipeline_metrics": {
    "vibe_fetch_ms": 1.2,
    "faiss_search_ms": 4.7,
    "db_fetch_ms": 12.1,
    "ranking_ms": 1.8,
    "postprocessing_ms": 0.9,
    "total_pipeline_ms": 20.7
  },
  "quality_metrics": {
    "intra_list_diversity": 0.387,
    "novelty_score": 0.412,
    "genre_spread": 4,
    "unique_artists": 6,
    "candidates_before_dedup": 30,
    "candidates_after_dedup": 6
  },
  "query_vector_preview": [0.12, -0.43, 0.87, 0.22, -0.11, 0.55, 0.09, -0.77],
  "vibe_steering_applied": true
}
```

### Telemetry

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/telemetry` | Ingest a behavioral event |

**Telemetry Event Schema:**
```json
{
  "user_id": 1,
  "track_id": 7,
  "event_type": "skip",
  "duration_ms": 4200,
  "listened_percentage": 0.07,
  "is_rapid_skip": true,
  "context": {
    "bpm": 93.4,
    "genre": "Electronic",
    "energy": 0.81,
    "valence": 0.74
  }
}
```

**Event Types:** `play`, `pause`, `heartbeat` (every 30s), `skip`, `track_completed`, `save`, `back`, `crossfade_start`

### Admin / ML Engineering

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/admin/stats` | FAISS size, track count, Redis keys, Kafka status |
| `GET` | `/api/v1/admin/tracks` | Full track list with all metadata & features |
| `GET` | `/api/v1/admin/tsne` | t-SNE 2D projection of all 128-d embeddings |
| `GET` | `/api/v1/admin/telemetry` | Last 100 behavioral events (ring buffer) |
| `GET` | `/api/v1/admin/redis` | All `vibe:*` keys and field values |
| `GET` | `/api/v1/admin/evaluation` | NDCG@10, Recall@10, MRR (offline evaluation) |
| `GET` | `/api/v1/admin/track/{id}/embedding` | Raw 128-d vector for a track |
| `GET` | `/api/v1/admin/audio/{id}` | Serves the raw audio file (FileResponse) |
| `POST` | `/api/v1/admin/simulate` | Fires 20 synthetic telemetry events |

### Explainability

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/tracks/{id}/explain` | Grad-CAM heatmap URL + recommendation reason text |
| `GET` | `/static/spectrograms/track_{id}.png` | Mel-spectrogram image |
| `GET` | `/static/gradcam/track_{id}.png` | Grad-CAM heatmap overlay |

---

## Frontend Dashboard — Module Breakdown

The frontend is a single `static/index.html` (~900 lines, zero build step) served directly from FastAPI. It uses Tailwind CSS and Chart.js from CDN with a dark violet theme (`#0f0d15` background, `#8B5CF6` accent).

### 🎵 Tab 1 — Player (User View)

- **Recommendation Grid:** Responsive card grid showing 6 recommended tracks with title, artist, genre badge, energy/valence bars, BPM, and spectrogram thumbnail.
- **"🧠 Why?" Explainability Drawer:** Per-track drawer displaying the Grad-CAM heatmap overlay and natural language explanation of why the model selected this track.
- **Persistent Bottom Player Bar:**
  - **Dual HTML5 Audio Engine:** Two `Audio` elements managed in an `audioCtx` object, enabling smooth crossfading between tracks without audio gaps.
  - **Customizable Crossfade:** 🔀 slider (0–12s). When the crossfade threshold is reached (e.g., 4s left in current track), the new track starts at 0% volume and fades in linearly while the current track fades out over N steps.
  - **Controls:** ⏮ Prev, ▶/⏸ Play/Pause, ⏭ Next (auto-fetches new recommendations when playlist is exhausted), Seekable progress bar, Volume slider, ❤️ Save.
  - **Dynamic Metadata:** Shows live BPM and genre (e.g., `"104 BPM • Hip-Hop"`) pulled from the recommendation response.

### 🎛️ Tab 2 — Vibe Studio (User View)

- **4 Range Sliders** (0.0–1.0, step 0.01): Energy, Valence, Acousticness, Danceability.
- **Radar Chart (Chart.js):** Updates in real time as sliders move—no button press needed.
- **Before/After Comparison:** Saves the current playlist before applying the vibe shift; displays both playlists side-by-side after the new recommendations arrive so the user can see the effect of their vector steering.
- **Vibe Steering Badge:** `vibe_steering_applied: true/false` indicator.

### 🔬 Tab 3 — ML Lab (Admin View)

- **System Overview Cards:** Total Tracks, FAISS Index Size, Redis Keys, Kafka Status (with live message count).
- **Pipeline Latency Waterfall (Chart.js Stacked Bar):** One segment per stage (Redis Vibe, FAISS Retrieval, PostgreSQL Fetch, D&W Ranking, Post-Processing). Updates on every recommendation call.
- **FAISS Vector Space (t-SNE Scatter):** Interactive Chart.js scatter plot projecting all 128-d embeddings into 2D, colored by genre. Hover tooltips show track title and artist.
- **Spectrogram Gallery:** Grid of mel-spectrogram PNGs (`static/spectrograms/track_{id}.png`). Hover reveals the Grad-CAM heatmap overlay.
- **128-D Embedding Inspector:** Select any track from a dropdown → Chart.js bar chart renders all 128 embedding dimensions (x-axis: dim 1–128, y-axis: activation value). Useful for understanding which frequency clusters drive similarity.

### 📊 Tab 4 — Data & Metrics (Admin View)

- **Quality Gauge Charts (Chart.js Doughnut):** 4 live gauges with center labels:
  - **Intra-List Diversity (ILD):** Average pairwise cosine distance.
  - **Novelty Score:** Long-tail exposure index.
  - **Genre Spread:** Number of distinct genres in the last recommendation batch.
  - **Artist Coverage:** Unique artists / batch size.
- **Offline Evaluation Scores Table:** Click "Compute" to trigger `GET /api/v1/admin/evaluation` — shows NDCG@10, Recall@10, and MRR with benchmark comparison badges.
- **Artist Deduplication Visualization:** Two-column side-by-side comparison — "Raw FAISS Candidates (30)" vs "After Dedup (6)" — with crossed-out removed duplicate artist tracks.
- **Live Telemetry Feed (auto-polls every 1s):** Scrolling table of last 100 events with track name, event type, listened percentage, and rich metadata column (`bpm:103 genre:Hip-Hop energy:0.8`).
- **Redis Vibe State Monitor:** Live table of all `vibe:*` keys showing each dimension for each user.
- **Simulate 20 Events Button:** Triggers `POST /api/v1/admin/simulate` → fires 20 synthetic telemetry events (play, skip, complete, save) → telemetry feed populates in real time.

---

## Recommendation Quality Metrics

Harmonix evaluates its recommendations across three categories, matching industry evaluation practice:

### 1. Offline Algorithmic Metrics (Historical Data)

Computed against all tracks using genre-based ground truth clustering (tracks of the same genre as the query are considered relevant):

| Metric | Description | Harmonix Score |
|---|---|---|
| **NDCG@10** | Normalized Discounted Cumulative Gain — rewards placing the most relevant tracks at the top of the list | ~0.31 |
| **Recall@10** | Of all relevant tracks, what fraction appear in the top-10 recommendations | ~0.37 |
| **MRR** | Mean Reciprocal Rank — how far down the list before the first relevant track appears | ~0.44 |

### 2. Online / Real-Time Diversity Metrics

Computed per-request after the post-processing stage:

| Metric | Computation | Interpretation |
|---|---|---|
| **Intra-List Diversity (ILD)** | Avg pairwise cosine distance across embedded tracks | Higher = more diverse batch |
| **Novelty** | `mean(track_id % 10 / 10.0)` | Higher = more long-tail, lesser-known content |
| **Genre Spread** | `len({t.genre for t in final_tracks})` | More genres = more variety |
| **Artist Coverage** | `len(unique_artists)` | Diversity of creators |

### 3. Implicit Behavioral Metrics (Telemetry)

Derived from actual user interaction events:

| Metric | Signal | Threshold |
|---|---|---|
| **Playthrough Rate** | `listened_percentage` | > 0.30 → positive vibe blend signal |
| **Rapid Skip Rate** | `is_rapid_skip` | `listened_percentage < 0.08` |
| **Track Completion** | `event_type == "track_completed"` | `listened_percentage >= 0.90` |
| **Heartbeat** | emitted every 30s of continuous playback | session health signal |

---

## Evaluation & Benchmarks

### Industry Algorithm Benchmarks Used as Reference

| Platform | Architecture | What Harmonix Implements |
|---|---|---|
| **Spotify** | Word2Vec (Track2Vec), 2T-HGNN | Embedding similarity in shared vector space |
| **YouTube Music** | Deep Candidate Generation + Deep Ranking (DNN) | Two-stage funnel: retrieval → scoring |
| **Gaana** | GraphSAGE (GNN co-listen graph) | Bipartite graph candidate retrieval concept |
| **Netflix/Meta** | FAISS IndexFlatIP, IndexHNSW | ANN search over dense embedding space |

### Standard Benchmark Datasets (for Future Full-Scale Training)

| Dataset | Size | Use Case |
|---|---|---|
| **Spotify Million Playlist Dataset (MPD)** | 1M playlists, 2M unique tracks | Automatic playlist continuation, Recall@K |
| **LFM-2b** | 2B events, 120K users, 15 years | Sequential taste modeling, fairness analysis |
| **Million Song Dataset (MSD)** | 1M tracks, rich audio features | Content-based audio analysis benchmarking |
| **Music4All** | Multimodal (audio + lyrics + tags) | Multimodal recommendation research |

---

## Algorithms & Research Foundation

### Core Algorithms Implemented

1. **FAISS ANN (Approximate Nearest Neighbor) Search**
   - Index type: `IndexFlatIP` (exact cosine similarity over L2-normalized vectors)
   - Query: biased user vector shifted by 4-dimensional vibe state
   - Retrieval: top `limit × 5` candidates, returned with distances

2. **Vibe State Alpha-Blending (Exponential Moving Average)**
   - Implicit signal: if `listened_percentage > 0.30`, blend track's audio features into user's vibe
   - Save signal: if `event_type == "save"`, stronger blend (`alpha = 0.20` vs `0.05`)

3. **ResNet-18 Audio Feature Extractor**
   - Input: `(1, 1, 128, 128)` mel-spectrogram tensor
   - Architecture: ResNet-18 with modified first conv layer (1-channel input), `AdaptiveAvgPool2d`, `Linear(512, 128)`
   - Output: L2-normalized 128-d embedding

4. **Grad-CAM (Gradient-weighted Class Activation Mapping)**
   - Applied to the final convolutional layer of ResNet-18
   - Generates heatmaps showing which frequency bands and temporal segments drove the model's embedding for a given track
   - Rendered as a color overlay on the mel-spectrogram

5. **Intra-List Diversity (ILD)**
   ```python
   def compute_ild(embeddings: list) -> float:
       mat = np.array(embeddings)
       norms = np.linalg.norm(mat, axis=1, keepdims=True)
       mat = mat / (norms + 1e-8)
       sim_matrix = np.dot(mat, mat.T)
       n = len(mat)
       mask = np.ones((n, n), dtype=bool)
       np.fill_diagonal(mask, False)
       distances = 1.0 - sim_matrix[mask]
       return float(np.mean(distances))
   ```

6. **t-SNE 2D Projection**
   - Applied to all stored 128-d FAISS embeddings on request
   - Cached after first computation; invalidated on database reseed
   - Rendered as a genre-colored scatter plot for visual cluster inspection

### Academic Research Foundation

| Paper | Authors | Relevance |
|---|---|---|
| *Deep content-based music recommendation* | van den Oord et al. (NeurIPS 2013) | CNN on mel-spectrograms, cold-start solution |
| *Deep Neural Networks for YouTube Recommendations* | Covington et al. (ACM RecSys 2016) | Two-stage funnel: candidate gen → precision ranking |
| *Item2Vec: Neural Item Embedding for Collaborative Filtering* | Barkan & Koenigstein (IEEE MLSP 2016) | Playlist-as-sentence embedding (Word2Vec on tracks) |
| *Inductive Representation Learning on Large Graphs (GraphSAGE)* | Hamilton et al. (NeurIPS 2017) | GNN for graph-based recommendation (Gaana's approach) |
| *The Million Playlist Dataset and the ACM RecSys Challenge 2018* | Chen et al. (ACM 2018) | Evaluation metrics: NDCG, R-Precision, Recall@K |
| *Billion-scale similarity search with GPUs* | Johnson, Douze & Jégou (IEEE Trans. 2019) | FAISS architecture and ANN indexing |
| *librosa: Audio and music signal analysis in python* | McFee et al. (SciPy 2015) | Feature extraction foundation |

---

## Project Structure

```
Harmonix/
│
├── backend/
│   ├── main.py                  # FastAPI app, lifespan, router registration
│   ├── database.py              # SQLAlchemy async models (User, Track, Playlist, ListeningEvent)
│   ├── faiss_index.py           # FAISS index management, warm_up, search_candidates, get_all_embeddings
│   ├── redis_client.py          # get/set vibe state, async Redis operations
│   ├── kafka_producer.py        # Kafka producer with in-memory fallback
│   ├── seed_data.py             # Full ML pipeline: audio → spectrogram → embedding → Grad-CAM → DB
│   ├── models.py                # Pydantic schemas
│   ├── requirements.txt         # All Python dependencies
│   ├── Dockerfile               # Container build spec
│   │
│   ├── routes/
│   │   ├── recommend.py         # POST /recommend, POST /vibe — two-stage funnel
│   │   ├── telemetry.py         # POST /telemetry — event ingestion + Redis vibe update
│   │   └── admin.py             # All admin/debug endpoints
│   │
│   ├── ml/
│   │   ├── feature_extractor.py # ResNet-18 AudioFeatureExtractor class
│   │   └── explainability.py    # Grad-CAM generation, GET /tracks/{id}/explain
│   │
│   ├── train_two_tower.py       # Two-Tower neural network training script
│   └── train_deep_wide.py       # Deep & Wide ranking model training script
│
├── static/
│   ├── index.html               # Single-page dashboard (~900 lines, zero build step)
│   ├── spectrograms/            # Generated: track_{id}.png mel-spectrograms
│   └── gradcam/                 # Generated: track_{id}.png Grad-CAM heatmap overlays
│
├── audio_samples/               # Drop your *.mp3, *.wav, *.m4a, *.mpeg files here
│                                # Dataset: 36 tracks from Internet Archive (~228 MB)
│
├── diagrams/                    # PlantUML source files (*.puml)
│   ├── architecture.puml
│   ├── er-diagram.puml
│   ├── sequence-diagram.puml
│   └── use-case.puml
│
├── project-report/              # LaTeX academic report
│   ├── main.tex
│   ├── coverpage.tex
│   ├── chapters/                # Chapter-1.tex through Chapter-11.tex
│   └── img/                     # Report figures and screenshots
│
├── Harmonix.md                  # Full project specification, research, architecture decisions
├── implementation_plan 1.md     # Initial implementation roadmap
└── README.md                    # This file
```

---

## Setup & Installation

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | Required for `match` syntax used in type hints |
| PostgreSQL | 14+ | Async driver: `asyncpg` |
| Redis | 7+ | Vibe state storage |
| FFMPEG | Latest | Required for M4A/MPEG audio format decoding via pydub |
| Kafka | 3.x (Optional) | Falls back to in-memory ring buffer if not running |

### Step 1 — Clone & Create Virtual Environment

```bash
git clone <your-repo-url>
cd Harmonix

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### Step 2 — Install Python Dependencies

```bash
pip install -r backend/requirements.txt
```

### Step 3 — Install FFMPEG (Windows)

```powershell
winget install Gyan.FFmpeg.Essentials --accept-source-agreements --accept-package-agreements
```

> FFMPEG is required for pydub to decode `.m4a` and `.mpeg` audio files. On Windows, this is the only external binary dependency.

### Step 4 — Start PostgreSQL & Redis

**Using Docker (recommended):**

```bash
docker run -d --name harmonix-postgres \
  -e POSTGRES_USER=harmonix_user \
  -e POSTGRES_PASSWORD=harmonix_password \
  -e POSTGRES_DB=harmonix \
  -p 5432:5432 postgres:15

docker run -d --name harmonix-redis \
  -p 6379:6379 redis:7
```

**Or use your local instances** — ensure the connection string matches:

```
postgresql+asyncpg://harmonix_user:harmonix_password@localhost/harmonix
```

### Step 5 — Add Your Audio Files

Place audio files in the `audio_samples/` directory:

```
audio_samples/
├── your_song.mp3
├── another_track.m4a
├── wav_file.wav
└── mpeg_track.mpeg
```

Supported formats: `.mp3`, `.wav`, `.m4a`, `.mpeg`

---

## Running the Application

### 1. Seed the Database (One-Time Setup)

```bash
python backend/seed_data.py
```

This runs the full ML pipeline:
- Scans `audio_samples/` for all audio files
- Decodes each with FFMPEG + pydub, loads with Librosa
- Generates mel-spectrogram PNGs → `static/spectrograms/`
- Extracts BPM, energy, valence, acousticness, danceability
- Runs ResNet-18 forward pass to produce 128-d embeddings
- Generates Grad-CAM heatmap PNGs → `static/gradcam/`
- Inserts all tracks into PostgreSQL with full metadata
- Builds FAISS index from all embeddings

### 2. Start the Server

```bash
uvicorn backend.main:app --reload
```

Server will start at `http://localhost:8000`.

On startup, the server:
1. Initializes PostgreSQL tables (`init_db()`)
2. Warms up the FAISS index from PostgreSQL (`warm_up_faiss()`)
3. Connects to Redis and Kafka (with graceful fallback)

### 3. Open the Dashboard

Navigate to `http://localhost:8000` in your browser.

**Interactive API Docs:** `http://localhost:8000/docs`

---

## Seeding the Database

The seed script is the entry point for the entire offline ML pipeline. It should be re-run whenever you add new audio files or want to rebuild the FAISS index.

```bash
# Full seed from audio_samples/
python backend/seed_data.py

# Expected output:
# [1/36] Processing: "Cheap Thrills" by Sia (genre: Electronic, BPM: 93.4)
# [2/36] Processing: "Iktara" by Amit Trivedi (genre: Classical, BPM: 68.2)
# ...
# ✅ All 36 tracks seeded.
# FAISS index size: 36 vectors
```

> **Note:** Re-seeding drops and recreates the tracks table. PostgreSQL user data and Redis vibe states persist unchanged.

---

## Testing

### Run the Integration Test Suite

```bash
pytest backend/test_integration.py -v
```

### Key Assertions

| Test | Assertion |
|---|---|
| `test_recommend_latency` | `total_pipeline_ms < 200` |
| `test_recommend_diversity` | `intra_list_diversity > 0.1` |
| `test_telemetry_ingestion` | Returns `{"status": "ok"}` |
| `test_audio_serving` | `GET /api/v1/admin/audio/1` returns audio bytes |
| `test_evaluation_endpoint` | `ndcg_at_10`, `recall_at_10`, `mrr` all in `[0.0, 1.0]` |
| `test_redis_vibe_update` | After telemetry with `pct > 0.30`, Redis hash updates |

### Manual Verification Checklist

After starting the server and seeding the database:

- [ ] **Player tab** loads with real track cards (not synthetic fallbacks)
- [ ] **Click any track card** → audio begins playing, player bar shows dynamic BPM & genre
- [ ] **Crossfade slider** → drag to 4s, let a track near its end → seamless audio transition
- [ ] **⏭ Next (at end of playlist)** → auto-fetches 6 new recommendations and continues
- [ ] **⏮ / ⏭** → prev/next navigation works; rapid skip (`< 8%`) logs `is_rapid_skip: true`
- [ ] **❤️ Save** → logs `save` event; stronger vibe blend applied
- [ ] **🧠 Why?** → explainability drawer opens with Grad-CAM image and text
- [ ] **Vibe Studio sliders** → radar chart updates live; Apply & Recommend shows before/after
- [ ] **ML Lab → Load t-SNE** → shows colored scatter plot by genre with track tooltips
- [ ] **ML Lab → Embedding Inspector** → select any track → 128-bar chart renders
- [ ] **Data & Metrics → Compute** → NDCG, Recall, MRR table populates
- [ ] **Data & Metrics → Simulate 20 Events** → telemetry feed populates in real time
- [ ] **Redis monitor** → shows vibe state after playing any track > 30%
- [ ] **Kafka stat** → shows `"connected (N msgs)"` and N increments each telemetry event

---

## Work Distribution

| Team Member | Responsibilities |
|---|---|
| **Person 1 — ML & Backend Engineer** | Audio preprocessing pipeline (Librosa, pydub, FFMPEG), ResNet-18 `AudioFeatureExtractor`, Grad-CAM generation, FAISS index management, `seed_data.py` ML pipeline, `GET /api/v1/admin/evaluation`, `GET /api/v1/admin/tsne`, offline evaluation metrics (NDCG, MRR, Recall) |
| **Person 2 — Data Engineer & Full-Stack** | FastAPI routing & lifespan, Redis vibe state management, Kafka producer, `POST /api/v1/telemetry` implicit feedback loop, dual-audio crossfade engine (JS), Chart.js visualizations (waterfall, gauges, scatter, radar), full `static/index.html` dashboard |

---

## Future Roadmap

### Near-Term (Phase 2)
- [ ] **True Two-Tower Model Training** — Replace heuristic vibe-shift with a trained neural retrieval model using the listening event log from PostgreSQL
- [ ] **Deep & Wide Ranking** — Deploy the trained `train_deep_wide.py` model for real contextual scoring (time of day, device type, recency)
- [ ] **Cold-Start Onboarding** — Interactive genre/artist selection modal for new users to seed their initial vibe vector

### Mid-Term (Phase 3)
- [ ] **GraphSAGE Co-Listen Graph** — Build a bipartite user-track interaction graph and train GraphSAGE for improved candidate recall
- [ ] **"Made for You" Daily Mixes** — K-Means clustering on user listening history to auto-generate "Deep Focus Mix", "Evening Acoustic Mix" playlists
- [ ] **Sequential Taste Modeling** — LSTM/Transformer on listening session logs (LFM-2b format) to predict the next track in a session

### Long-Term (Phase 4)
- [ ] **Real Kafka Cluster** — Replace in-memory fallback with distributed Kafka for multi-user concurrent event ingestion
- [ ] **Kubernetes Deployment** — Containerize all services and deploy on k8s for horizontal scaling
- [ ] **A/B Testing Framework** — Implement experiment management to compare Two-Tower vs GraphSAGE candidate quality
- [ ] **Taste Blend (Social)** — Vector space overlap between two users to generate a shared playlist
- [ ] **LLM Playlist Titles** — Use an LLM to generate contextual, human-readable playlist descriptions from the cluster's audio feature centroid

---

## References

1. van den Oord, A., Dieleman, S., & Schrauwen, B. (2013). *Deep content-based music recommendation*. NeurIPS.
2. Covington, P., Adams, J., & Sargin, E. (2016). *Deep neural networks for YouTube recommendations*. ACM RecSys.
3. Barkan, O., & Koenigstein, N. (2016). *Item2Vec: Neural item embedding for collaborative filtering*. IEEE MLSP.
4. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive representation learning on large graphs (GraphSAGE)*. NeurIPS.
5. Chen, C. W., Lamere, P., Schedl, M., & Zamani, H. (2018). *The million playlist dataset and the ACM RecSys challenge 2018*. ACM RecSys.
6. Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity search with GPUs*. IEEE Transactions on Big Data.
7. McFee, B., et al. (2015). *librosa: Audio and music signal analysis in Python*. SciPy Proceedings.
8. Oard, D. W., & Kim, J. (1998). *Implicit feedback for recommender systems*. AAAI Workshop on Recommender Systems.

---

<div align="center">

**Harmonix** · Built by Apaar Mathur & ❤️

</div>