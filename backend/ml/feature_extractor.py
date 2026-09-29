import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

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
        
        # Grad-CAM specific attributes
        self.gradients = None
        self.activations = None
        
        # Register hooks for Grad-CAM on layer4
        self.resnet.layer4.register_forward_hook(self._forward_hook)
        self.resnet.layer4.register_full_backward_hook(self._backward_hook)
        
    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]
        
    def forward(self, x):
        # x expected shape: (batch, 1, 128, 128) - mel spectrograms
        embeddings = self.resnet(x)
        # L2 Normalize
        return nn.functional.normalize(embeddings, p=2, dim=1)
        
    def generate_gradcam(self, x):
        """Generates a Grad-CAM heatmap for a single batch input (1, 1, 128, 128)."""
        self.eval()
        self.zero_grad()
        
        # Forward pass
        out = self.resnet(x)
        
        # We want to explain the most active feature dimension for this track
        target_idx = torch.argmax(out[0])
        out[0, target_idx].backward(retain_graph=True)
        
        # Pull gradients and activations
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()[0]
        
        # Weight activations by the gradients
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]
            
        # Average over channels and apply ReLU
        heatmap = torch.mean(activations, dim=0).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        heatmap_max = np.max(heatmap)
        if heatmap_max == 0:
            return heatmap
        
        heatmap /= heatmap_max
        return heatmap
