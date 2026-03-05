from torch import nn
import torch

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int=3, patch_size: int=16, embedding_dim: int=768):
        super().__init__()
        self.patcher = nn.Conv2d(in_channels=in_channels, 
                                 out_channels=embedding_dim, 
                                 kernel_size=patch_size, 
                                 stride=patch_size, 
                                 padding=0)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
    def forward(self, x):
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0, 2, 1)
    
