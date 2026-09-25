from fastapi import APIRouter, BackgroundTasks, status
from pydantic import BaseModel
from typing import Optional
import uuid
from backend.kafka_producer import send_telemetry
from datetime import datetime, timezone

router = APIRouter()

class Context(BaseModel):
    source: Optional[str] = None
    device: Optional[str] = None
    sliders: Optional[dict] = None

class TelemetryEvent(BaseModel):
    user_id: int
    track_id: int
    event_type: str
    duration_ms: int
    context: Optional[Context] = None
    timestamp: datetime = datetime.now(timezone.utc)

@router.post("/telemetry", status_code=status.HTTP_202_ACCEPTED)
async def ingest_telemetry(event: TelemetryEvent, background_tasks: BackgroundTasks):
    event_dict = event.model_dump()
    event_dict['timestamp'] = event_dict['timestamp'].isoformat()
    background_tasks.add_task(send_telemetry, event_dict)
    return {"status": "queued", "event_id": str(uuid.uuid4())}
