import torch
from torch import nn

# ==========================================
# 1. MULTI-HEAD ATTENTION
# ==========================================
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim:int=512, num_heads:int=8, attn_dropout:float=0.1):
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

# ==========================================
# 2. MLP BLOCK (Feed-Forward)
# ==========================================
class MLPBlock(nn.Module):
    def __init__(self, embedding_dim:int=512, mlp_dim:int=1024, dropout:float=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_dim),
            nn.ReLU(), # Dùng ReLU theo bài báo Optimized ViT
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_dim, out_features=embedding_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.mlp(x_norm)

# ==========================================
# 3. TRANSFORMER ENCODER BLOCK
# ==========================================
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim: int=512, num_heads: int=8, mlp_dim: int=1024, mlp_dropout: float=0.1, attn_dropout: float=0.1):
        super().__init__()
        self.msa = MultiHeadAttention(embedding_dim, num_heads, attn_dropout)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, mlp_dropout)
        
    def forward(self, x):
        # Kiến trúc Residual Connection
        x = x + self.msa(x)
        x = x + self.mlp(x)
        return x

# ==========================================
# 4. HÀM CHẠY TEST
# ==========================================
def main():
    print("\n" + "="*50)
    print("🧠 BẮT ĐẦU TEST: TRANSFORMER ENCODER BLOCK")
    print("="*50)
    
    # Cấu hình giả lập
    BATCH_SIZE = 32
    NUM_TOKENS = 197 # 196 patches + 1 [CLS] token
    EMBEDDING_DIM = 512
    
    # 1. Tạo dữ liệu đầu vào giả lập
    # Tưởng tượng đây là dữ liệu đã đi qua PatchEmbedding và Positional Encoding
    print("[INFO] Đang tạo dữ liệu đầu vào giả lập...")
    dummy_input = torch.randn(BATCH_SIZE, NUM_TOKENS, EMBEDDING_DIM)
    
    print(f"1. Kích thước Tensor đầu vào: {list(dummy_input.shape)}")
    
    # 2. Khởi tạo khối Transformer
    print("[INFO] Đang khởi tạo TransformerEncoderBlock với thông số Optimized ViT...")
    transformer_block = TransformerEncoderBlock(
        embedding_dim=512, 
        num_heads=8, 
        mlp_dim=1024
    )
    
    # 3. Chạy dữ liệu qua khối Transformer
    print("[INFO] Đang đẩy dữ liệu qua mạng...")
    output = transformer_block(dummy_input)
    
    print(f"2. Kích thước Tensor đầu ra:  {list(output.shape)}")
    
    # 4. Kiểm tra toán học
    print("\n" + "-"*50)
    print("--- KIỂM TRA TOÁN HỌC ---")
    print("Đặc điểm của Transformer Block là đầu vào và đầu ra phải Y HỆT NHAU về mặt kích thước (để có thể xếp chồng nhiều lớp lên nhau).")
    
    if dummy_input.shape == output.shape:
        print("\n✅ PASSED! Kích thước không bị thay đổi. Đường nối tắt (Residual) hoạt động hoàn hảo!")
    else:
        print("\n❌ FAILED! Kích thước bị sai lệch sau khi qua Transformer Block.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()