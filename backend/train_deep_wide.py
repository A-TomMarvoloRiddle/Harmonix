import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.ml.ranking_models import DeepWideModel
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
    
    track_embeddings = [t.embedding for t in tracks if t.embedding]
    track_tensor = torch.tensor(track_embeddings, dtype=torch.float32)
    num_tracks = track_tensor.size(0)
    
    # 1. Synthesize Ranking Dataset
    # Features: User Vector (128), Track Vector (128), Context (4: time_of_day, device, volume, prev_skip)
    num_samples = 1000
    
    X_users = []
    X_tracks = []
    X_context = []
    Y_labels = []
    
    for _ in range(num_samples):
        u_vec = torch.randn(128) # Simulated user
        t_vec = track_tensor[torch.randint(0, num_tracks, (1,))].squeeze(0)
        
        # Context: [Time(0-1), Mobile(0/1), Volume(0-1), PrevSkip(0/1)]
        ctx = torch.tensor([torch.rand(1).item(), float(torch.rand(1)>0.5), torch.rand(1).item(), float(torch.rand(1)>0.8)])
        
        # Simulated heuristic label: if context is prev_skip=1, lower prob. If vectors align, higher prob.
        alignment = torch.dot(u_vec/torch.norm(u_vec), t_vec/torch.norm(t_vec))
        prob = 0.5 + 0.3 * alignment.item() - 0.2 * ctx[3].item()
        
        label = 1.0 if prob > 0.5 else 0.0
        
        X_users.append(u_vec)
        X_tracks.append(t_vec)
        X_context.append(ctx)
        Y_labels.append(torch.tensor([label]))
        
    X_users = torch.stack(X_users)
    X_tracks = torch.stack(X_tracks)
    X_context = torch.stack(X_context)
    Y_labels = torch.stack(Y_labels)
    
    print("Initializing Deep & Wide Model...")
    model = DeepWideModel(embedding_dim=128, context_dim=4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    epochs = 30
    batch_size = 64
    
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(num_samples)
        total_loss = 0.0
        
        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i+batch_size]
            u_b = X_users[indices]
            t_b = X_tracks[indices]
            c_b = X_context[indices]
            y_b = Y_labels[indices]
            
            optimizer.zero_grad()
            preds = model(u_b, t_b, c_b)
            loss = criterion(preds, y_b)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | BCE Loss: {total_loss:.4f}")
        
    os.makedirs("backend/weights", exist_ok=True)
    torch.save(model.state_dict(), "backend/weights/deep_wide.pth")
    print("Saved trained weights to backend/weights/deep_wide.pth")

if __name__ == "__main__":
    asyncio.run(main())
