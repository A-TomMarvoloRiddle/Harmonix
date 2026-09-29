from fastapi import APIRouter, BackgroundTasks, status, Depends
from pydantic import BaseModel
from typing import Optional
import uuid
from backend.kafka_producer import send_telemetry
from datetime import datetime, timezone

router = APIRouter()

class TelemetryEvent(BaseModel):
    user_id: int
    track_id: int
    event_type: str
    duration_ms: int
    listened_percentage: float = 0.0
    is_rapid_skip: bool = False
    context: Optional[dict] = None
    timestamp: datetime = datetime.now(timezone.utc)

from backend.routes.admin import telemetry_buffer, stats_data
from backend.redis_client import get_vibe_studio_state, set_vibe_studio_state
from backend.database import async_session, Track
from sqlalchemy.future import select

@router.post("/telemetry", status_code=status.HTTP_202_ACCEPTED)
async def ingest_telemetry(event: TelemetryEvent, background_tasks: BackgroundTasks):
    event_dict = event.model_dump()
    event_dict['timestamp'] = event_dict['timestamp'].isoformat()
    
    # Append to buffer for admin UI
    telemetry_buffer.append(event_dict)
    stats_data["kafka_msg_count"] += 1
    
    # Implicit Feedback: Update Redis Vibe if track heavily listened
    if (event.event_type in ["track_completed", "pause", "skip"] and event.listened_percentage > 0.3) or event.event_type == "save":
        async with async_session() as session:
            result = await session.execute(select(Track).where(Track.id == event.track_id))
            track = result.scalars().first()
            if track:
                vibe = await get_vibe_studio_state(str(event.user_id))
                if not vibe:
                    vibe = {"energy": 0.5, "valence": 0.5, "acousticness": 0.5, "danceability": 0.5}
                # Blend 90% old vibe with 10% this track's vibe (or 20% for save)
                alpha = 0.2 if event.event_type == "save" else 0.05
                vibe["energy"] = float(vibe.get("energy", 0.5)) * (1 - alpha) + track.energy * alpha
                vibe["valence"] = float(vibe.get("valence", 0.5)) * (1 - alpha) + track.valence * alpha
                vibe["acousticness"] = float(vibe.get("acousticness", 0.5)) * (1 - alpha) + track.acousticness * alpha
                vibe["danceability"] = float(vibe.get("danceability", 0.5)) * (1 - alpha) + track.danceability * alpha
                await set_vibe_studio_state(str(event.user_id), vibe)
    
    # Send to Kafka (or mock)
    background_tasks.add_task(send_telemetry, event_dict)
    return {"status": "queued", "event_id": str(uuid.uuid4())}
