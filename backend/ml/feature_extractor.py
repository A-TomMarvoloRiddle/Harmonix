import torch
import torch.nn as nn
import torchvision.models as models

class AudioFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # Load pre-trained ResNet-18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify the first conv layer to accept 1-channel (grayscale mel-spectrogram)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Replace the final classification head with a projection to embedding_dim
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
        
    def forward(self, x):
        # x expected shape: (batch, 1, 128, 128) - mel spectrograms
        embeddings = self.resnet(x)
        # L2 Normalize
        return nn.functional.normalize(embeddings, p=2, dim=1)
