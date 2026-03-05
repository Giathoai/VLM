from torch import nn
import torch

from models.transformer import TransformerEncoderBlock
from models.patch_embed import PatchEmbedding
class VIT(nn.Module):
    def __init__(self, 
                 image_size: int=224, 
                 patch_size: int=16, 
                 in_channels: int=3,
                 embedding_dim: int=512, 
                 num_heads: int=8, 
                 mlp_dim: int=1024, 
                 num_layers: int=6, 
                 mlp_dropout: float=0.1, 
                 attn_dropout: float=0.1,
                 embed_dropout: float=0.1,
                 num_classes: int=6):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim), requires_grad=True)
        self.pos_embedding = nn.Parameter(data=self._get_sinusoid_encoding(self.num_patches + 1, embedding_dim), requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=embed_dropout)
        self.patch_embedding = PatchEmbedding(in_channels, 
                                              patch_size, 
                                              embedding_dim)
        self.transformers = nn.Sequential(*[TransformerEncoderBlock(embedding_dim, 
                                                                    num_heads, 
                                                                    mlp_dim, 
                                                                    mlp_dropout, 
                                                                    attn_dropout) for _ in range(num_layers)])

        self.classifier = nn.Sequential(nn.LayerNorm(embedding_dim), 
                                        nn.Linear(embedding_dim, num_classes))
    def _get_sinusoid_encoding(self, num_tokens, embedding_dim):
        pos = torch.arange(num_tokens, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        pos_embed = torch.zeros(1, num_tokens, embedding_dim)
        pos_embed[0, :, 0::2] = torch.sin(pos * div_term)
        pos_embed[0, :, 1::2] = torch.cos(pos * div_term)
        return pos_embed
    
    def forward(self, x):
        batch_size = x.shape[0]
        cls_tokens = self.class_embedding.expand(batch_size, -1, -1)
        x = self.patch_embedding(x)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.embedding_dropout(x)
        x = self.transformers(x)
        cls_output = x[:, 0]
        out = self.classifier(cls_output)
        return out
    def get_features(self, x):
        batch_size = x.shape[0]
        cls_tokens = self.class_embedding.expand(batch_size, -1, -1)
        
        x = self.patch_embedding(x)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.embedding_dropout(x)
        
        x = self.transformers(x)
        
        return x 