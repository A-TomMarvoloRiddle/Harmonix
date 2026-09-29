import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    """
    Two-Tower metric learning model mapping User vectors and Track vectors 
    into a shared embedding space for rapid ANN retrieval.
    """
    def __init__(self, embedding_dim=128):
        super().__init__()
        # In a real scenario, this would have user ID embedding tables or sequence models.
        # Here we just refine the existing track embedding and a naive user profile vector.
        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        self.track_tower = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        
    def forward(self, user_features, track_features):
        u_emb = self.user_tower(user_features)
        t_emb = self.track_tower(track_features)
        
        # Normalize for inner product (cosine similarity)
        u_emb = nn.functional.normalize(u_emb, p=2, dim=1)
        t_emb = nn.functional.normalize(t_emb, p=2, dim=1)
        
        return u_emb, t_emb

class DeepWideModel(nn.Module):
    """
    Deep & Wide contextual ranking model.
    Takes [user_vector, track_vector, context_features] and outputs engagement probability.
    """
    def __init__(self, embedding_dim=128, context_dim=4):
        super().__init__()
        
        input_dim = (embedding_dim * 2) + context_dim
        
        # Deep component
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Wide component (direct linear connection for memorization)
        self.wide = nn.Linear(input_dim, 1)
        
    def forward(self, user_features, track_features, context_features):
        x = torch.cat([user_features, track_features, context_features], dim=1)
        deep_out = self.deep(x)
        wide_out = self.wide(x)
        return torch.sigmoid(deep_out + wide_out)
