import json
import logging
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

logger = logging.getLogger(__name__)
_producer = None

def get_producer():
    global _producer
    if _producer is None:
        try:
            _producer = KafkaProducer(
                bootstrap_servers=['localhost:9092'],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: str(k).encode('utf-8') if k else None
            )
        except NoBrokersAvailable:
            logger.warning("Kafka broker unavailable. Telemetry will be mocked locally.")
    return _producer

def send_telemetry(event: dict):
    producer = get_producer()
    if producer:
        try:
            producer.send('harmonix-telemetry-events', key=event.get("user_id"), value=event)
        except Exception as e:
            logger.error(f"Failed to send telemetry to Kafka: {e}")
    else:
        logger.info(f"[MOCK KAFKA] Telemetry Event: {event}")

def close_producer():
    if _producer:
        _producer.close()
