import torch
import torch.nn as nn

class TwoTowerNetwork(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=128):
        super().__init__()
        # User Tower
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        
        # Item Tower
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.item_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        
    def forward(self, user_ids, item_ids):
        u_emb = self.user_mlp(self.user_embedding(user_ids))
        i_emb = self.item_mlp(self.item_embedding(item_ids))
        
        # L2 Normalize
        u_emb = nn.functional.normalize(u_emb, p=2, dim=1)
        i_emb = nn.functional.normalize(i_emb, p=2, dim=1)
        
        # Return inner product
        return (u_emb * i_emb).sum(dim=1)

class DeepWideNetwork(nn.Module):
    def __init__(self, continuous_features_dim=3, user_dim=128):
        super().__init__()
        # Deep Component
        self.deep = nn.Sequential(
            nn.Linear(continuous_features_dim + user_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Wide Component (Simulated by linear layer on continuous features)
        self.wide = nn.Linear(continuous_features_dim, 1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_vectors, continuous_features):
        deep_out = self.deep(torch.cat([user_vectors, continuous_features], dim=1))
        wide_out = self.wide(continuous_features)
        
        return self.sigmoid(deep_out + wide_out)
