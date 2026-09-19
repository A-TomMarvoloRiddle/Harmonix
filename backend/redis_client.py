import redis.asyncio as redis

# Create a connection pool for sub-millisecond caching
pool = redis.ConnectionPool.from_url("redis://localhost:6379", decode_responses=True)
redis_client = redis.Redis(connection_pool=pool)

async def get_vibe_studio_state(user_id: str) -> dict:
    return await redis_client.hgetall(f"vibe:{user_id}")

async def set_vibe_studio_state(user_id: str, state: dict):
    await redis_client.hset(f"vibe:{user_id}", mapping=state)
