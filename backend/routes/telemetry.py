from fastapi import APIRouter, BackgroundTasks, status, Depends
from pydantic import BaseModel, Field
from typing import Optional
import uuid
import asyncio
from backend.kafka_producer import send_telemetry
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class TelemetryEvent(BaseModel):
    user_id: int
    track_id: int
    event_type: str
    duration_ms: Optional[int] = 0
    listened_percentage: Optional[float] = 0.0
    is_rapid_skip: bool = False
    context: Optional[dict] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

from backend.routes.admin import telemetry_buffer, stats_data
from backend.redis_client import redis_client
from backend.database import async_session, Track
from sqlalchemy.future import select

async def update_vibe_state(user_id: str, track: Track, alpha: float):
    # Lua script for atomic fetch, blend, and store
    lua_script = """
    local vibe = redis.call('HGETALL', KEYS[1])
    local dict = {}
    if #vibe == 0 then
        dict['energy'] = 0.5
        dict['valence'] = 0.5
        dict['acousticness'] = 0.5
        dict['danceability'] = 0.5
    else
        for i=1, #vibe, 2 do
            dict[vibe[i]] = tonumber(vibe[i+1]) or 0.5
        end
    end
    
    local alpha = tonumber(ARGV[1])
    local new_energy = dict['energy'] * (1 - alpha) + tonumber(ARGV[2]) * alpha
    local new_valence = dict['valence'] * (1 - alpha) + tonumber(ARGV[3]) * alpha
    local new_acousticness = dict['acousticness'] * (1 - alpha) + tonumber(ARGV[4]) * alpha
    local new_danceability = dict['danceability'] * (1 - alpha) + tonumber(ARGV[5]) * alpha
    
    redis.call('HSET', KEYS[1], 
        'energy', new_energy,
        'valence', new_valence,
        'acousticness', new_acousticness,
        'danceability', new_danceability
    )
    return 1
    """
    await redis_client.eval(
        lua_script, 1, f"vibe:{user_id}", float(alpha), 
        float(track.energy), float(track.valence), float(track.acousticness), float(track.danceability)
    )

@router.post("/telemetry", status_code=status.HTTP_202_ACCEPTED)
async def ingest_telemetry(event: TelemetryEvent, background_tasks: BackgroundTasks):
    if event.listened_percentage is None: event.listened_percentage = 0.0
    if event.duration_ms is None: event.duration_ms = 0

    event_dict = event.model_dump()
    event_dict['timestamp'] = event_dict['timestamp'].isoformat()
    
    # Append to buffer for admin UI
    telemetry_buffer.append(event_dict)
    stats_data["kafka_msg_count"] += 1
    
    # Implicit Feedback: Update Redis Vibe aggressively for the demo on any interaction
    if event.event_type in ["track_completed", "pause", "skip", "save"]:
        try:
            async with async_session() as session:
                result = await session.execute(select(Track).where(Track.id == event.track_id))
                track = result.scalars().first()
                if track:
                    # Larger alpha for save (0.3), negative for rapid skip (-0.1), small positive for normal interactions (0.1)
                    if event.event_type == "save":
                        alpha = 0.3
                    elif event.is_rapid_skip:
                        alpha = -0.1
                    else:
                        alpha = 0.1
                    await update_vibe_state(str(event.user_id), track, alpha)
        except Exception as e:
            logger.error(f"Failed to update vibe state: {e}")
    
    # Send to Kafka (or async fallback queue)
    asyncio.create_task(send_telemetry(event_dict))
    
    return {"status": "queued", "event_id": str(uuid.uuid4())}
