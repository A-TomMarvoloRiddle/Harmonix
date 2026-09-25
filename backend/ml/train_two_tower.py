import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from backend.ml.models import TwoTowerNetwork

def train():
    print("Mocking training Two Tower Network...")
    model = TwoTowerNetwork(num_users=1000, num_items=5000)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Mock data
    user_ids = torch.randint(0, 1000, (32,))
    item_ids = torch.randint(0, 5000, (32,))
    labels = torch.randint(0, 2, (32,)).float()
    
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(2):
        optimizer.zero_grad()
        logits = model(user_ids, item_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
