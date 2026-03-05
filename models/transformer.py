from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim:int=768,num_heads:int=8, attn_dropout:float=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, 
                                                    num_heads=num_heads, 
                                                    dropout=attn_dropout,          
                                                    batch_first=True)
    def forward(self, x):
        x_norm = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x_norm, key=x_norm, value=x_norm, need_weights=False)
        return attn_output
    
class MLPBlock(nn.Module):
    def __init__(self, embedding_dim:int=768, mlp_dim:int=1024, dropout:float=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_dim, out_features=embedding_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.mlp(x_norm)
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim: int=768, num_heads: int=8, mlp_dim: int=1024, mlp_dropout: float=0.1, attn_dropout: float=0.1):
        super().__init__()
        self.msa = MultiHeadAttention(embedding_dim, num_heads, attn_dropout)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, mlp_dropout)
    def forward(self, x):
        x = x + self.msa(x)
        x = x + self.mlp(x)
        return x