import asyncio
import sys
from pathlib import Path

# Add project root to sys.path so it works when run either directly or as a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.database import engine, Base, async_session, User, Track
import numpy as np
from backend.faiss_index import add_embeddings, EMBEDDING_DIM

async def seed():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with async_session() as session:
        # Add mock user
        user = User(username="test_user", preferences={"genre": "electronic"})
        session.add(user)
        
        # Add mock tracks
        for i in range(50):
            track = Track(
                title=f"Mock Track {i}",
                artist=f"Artist {i % 10}",
            )
            session.add(track)
            
        await session.commit()

    # Seed FAISS
    mock_embeddings = np.random.randn(50, EMBEDDING_DIM).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(mock_embeddings, axis=1, keepdims=True)
    mock_embeddings = mock_embeddings / norms
    add_embeddings(mock_embeddings)
    
    print("Database and FAISS seeded successfully!")

if __name__ == "__main__":
    asyncio.run(seed())
