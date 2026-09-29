from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.database import init_db
from backend.kafka_producer import close_producer, init_kafka
import logging

from backend.routes.telemetry import router as telemetry_router
from backend.routes.recommend import router as recommend_router
from backend.ml.explainability import router as explain_router
from backend.routes.admin import router as admin_router
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

logging.basicConfig(level=logging.INFO)

from backend.faiss_index import warm_up_faiss

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    # Warm up FAISS index from Postgres
    await warm_up_faiss()
    # Init Kafka producer or async fallback
    init_kafka()
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

# Mount static files
import os
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(telemetry_router, prefix="/api/v1", tags=["Telemetry"])
app.include_router(recommend_router, prefix="/api/v1", tags=["Recommendations"])
app.include_router(explain_router, prefix="/api/v1", tags=["Explainability"])
app.include_router(admin_router, prefix="/api/v1/admin", tags=["Admin"])

@app.get("/", include_in_schema=False)
async def serve_dashboard():
    return FileResponse("static/index.html")

@app.get("/health", tags=["Health"])
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
