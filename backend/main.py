from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.database import init_db
from backend.kafka_producer import close_producer
import logging

from backend.routes.telemetry import router as telemetry_router
from backend.routes.recommend import router as recommend_router
from backend.ml.explainability import router as explain_router

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    # Mock FAISS warming in faiss_index
    yield
    # Shutdown
    close_producer()

app = FastAPI(title="Harmonix API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(telemetry_router, prefix="/api/v1", tags=["Telemetry"])
app.include_router(recommend_router, prefix="/api/v1", tags=["Recommendations"])
app.include_router(explain_router, prefix="/api/v1", tags=["Explainability"])

@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "online",
        "service": "Harmonix Two-Stage Music Recommendation Engine",
        "version": "1.0.0",
        "docs_url": "http://127.0.0.1:8000/docs",
        "endpoints": {
            "recommend": "POST /api/v1/recommend",
            "vibe": "POST /api/v1/vibe",
            "telemetry": "POST /api/v1/telemetry",
            "explain": "GET /api/v1/tracks/{track_id}/explain"
        }
    }
