import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.ml.ranking_models import TwoTowerModel
from backend.database import async_session, Track
import asyncio
from sqlalchemy.future import select

async def main():
    print("Loading tracks from database...")
    async with async_session() as session:
        result = await session.execute(select(Track))
        tracks = result.scalars().all()
        
    if not tracks or not tracks[0].embedding:
        print("Error: No tracks with embeddings found. Run seed_data.py first.")
        return
        
    print(f"Loaded {len(tracks)} tracks.")
    
    # 1. Create a mock synthetic history dataset for training
    # We'll simulate 500 positive pairs and 500 negative pairs
    
    track_embeddings = [t.embedding for t in tracks if t.embedding]
    track_tensor = torch.tensor(track_embeddings, dtype=torch.float32)
    
    num_tracks = track_tensor.size(0)
    
    # Simulating a user vector as the average of 3 random tracks they "liked"
    users = []
    for _ in range(200):
        indices = torch.randperm(num_tracks)[:3]
        user_vec = torch.mean(track_tensor[indices], dim=0)
        users.append(user_vec)
        
    users_tensor = torch.stack(users)
    
    print("Initializing Two-Tower Model...")
    model = TwoTowerModel(embedding_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Simple Contrastive Loss (InfoNCE approx)
    criterion = nn.MSELoss() 
    
    print("Training loop starting...")
    epochs = 25
    batch_size = 32
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        
        # Create random positive/negative batches
        for _ in range(num_tracks // batch_size + 1):
            # Positives
            u_idx = torch.randint(0, len(users), (batch_size,))
            t_idx_pos = torch.randint(0, num_tracks, (batch_size,))
            
            u_batch = users_tensor[u_idx]
            t_pos = track_tensor[t_idx_pos]
            
            optimizer.zero_grad()
            u_emb, t_emb = model(u_batch, t_pos)
            
            # Maximize similarity for positives (dot product -> 1.0)
            sim_pos = torch.sum(u_emb * t_emb, dim=1)
            loss = criterion(sim_pos, torch.ones_like(sim_pos))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")
        
    os.makedirs("backend/weights", exist_ok=True)
    torch.save(model.state_dict(), "backend/weights/two_tower.pth")
    print("Saved trained weights to backend/weights/two_tower.pth")

if __name__ == "__main__":
    asyncio.run(main())
