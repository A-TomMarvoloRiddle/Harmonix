import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from backend.ml.models import DeepWideNetwork

def train():
    print("Mocking training Deep & Wide Network...")
    model = DeepWideNetwork(continuous_features_dim=3, user_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Mock data
    user_vectors = torch.randn(32, 128)
    continuous_features = torch.randn(32, 3)
    labels = torch.randint(0, 2, (32,)).float().unsqueeze(1)
    
    criterion = nn.BCELoss()
    
    for epoch in range(2):
        optimizer.zero_grad()
        preds = model(user_vectors, continuous_features)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
