import json
import logging
import asyncio
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

logger = logging.getLogger(__name__)
_producer = None
_fallback_queue = None

async def fallback_worker():
    while True:
        event = await _fallback_queue.get()
        logger.info(f"[FALLBACK QUEUE] Processed Telemetry Event: {event.get('event_type')} for User {event.get('user_id')}")
        _fallback_queue.task_done()

def init_kafka():
    global _producer, _fallback_queue
    try:
        _producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: str(k).encode('utf-8') if k else None
        )
        logger.info("[KAFKA BROKER] Successfully connected to localhost:9092")
    except NoBrokersAvailable:
        logger.warning("[KAFKA BROKER] Unavailable. Initializing robust async fallback queue.")
        _fallback_queue = asyncio.Queue()
        asyncio.create_task(fallback_worker())

def get_producer():
    # Helper to lazily initialize if not called explicitly, though lifespan should call it
    global _producer, _fallback_queue
    if _producer is None and _fallback_queue is None:
        init_kafka()
    return _producer

async def send_telemetry(event: dict):
    producer = get_producer()
    if producer:
        try:
            producer.send('harmonix-telemetry-events', key=event.get("user_id"), value=event)
            logger.info(f"[KAFKA BROKER] Sent event: {event.get('event_type')}")
        except Exception as e:
            logger.error(f"[KAFKA BROKER] Send failed: {e}")
    elif _fallback_queue:
        await _fallback_queue.put(event)

def close_producer():
    global _producer
    if _producer:
        _producer.close()
