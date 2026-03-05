import sys
import os
import torch
from torch import nn
import math

# --- Setup đường dẫn để import dataloaders ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import create_dataloaders (nếu thư mục data không có, code vẫn tự tạo Dummy Data)
try:
    from dataloaders.dataset import create_dataloaders
except ImportError:
    create_dataloaders = None

# ==========================================
# 1. PATCH EMBEDDING
# ==========================================
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int=3, patch_size: int=16, embedding_dim: int=512):
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

# ==========================================
# 2. OPTIMIZED VIT (Giai đoạn 1: Chuẩn bị Input)
# ==========================================
class OptimizedViT(nn.Module):
    def __init__(self, in_channels: int=3, patch_size: int=16, embedding_dim: int=512, img_size: int=224):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels=in_channels, 
                                              patch_size=patch_size, 
                                              embedding_dim=embedding_dim)
        
        # [CLS] Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim), requires_grad=True)
        
        # Positional Encoding (Sine/Cosine)
        num_patches = (img_size // patch_size) ** 2 
        num_tokens = num_patches + 1
        self.pos_embedding = nn.Parameter(
            self._get_sinusoid_encoding(num_tokens, embedding_dim), 
            requires_grad=False # KHÔNG học, giữ cố định theo công thức
        )

    def _get_sinusoid_encoding(self, num_tokens, dim):
        position = torch.arange(num_tokens, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pos_embed = torch.zeros(1, num_tokens, dim)
        pos_embed[0, :, 0::2] = torch.sin(position * div_term) 
        pos_embed[0, :, 1::2] = torch.cos(position * div_term) 
        return pos_embed

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Bước 1: Patching
        x = self.patch_embedding(x) 
        
        # Bước 2: Thêm [CLS] Token
        cls_token_expanded = self.cls_token.expand(batch_size, -1, -1) 
        x = torch.cat((cls_token_expanded, x), dim=1)
        
        # Bước 3: Cộng Positional Encoding
        x = x + self.pos_embedding
        
        return x

# ==========================================
# 3. HÀM CHẠY TEST
# ==========================================
def main():
    print("\n" + "="*50)
    print("🚀 BẮT ĐẦU TEST: PREPARATION CHO TRANSFORMER ENCODER")
    print("="*50)
    
    # [cite_start]Cấu hình chuẩn của Optimized ViT [cite: 552]
    DATA_DIR = os.path.join(project_root, "data")
    IMAGE_SIZE = 224
    PATCH_SIZE = 16
    EMBEDDING_DIM = 512
    BATCH_SIZE = 32

    # Tự động lấy dữ liệu hoặc tạo dữ liệu giả lập
    train_path = os.path.join(DATA_DIR, 'Train')
    if os.path.exists(train_path) and create_dataloaders is not None:
        print(f"[INFO] Load dữ liệu thật từ: {DATA_DIR}")
        train_loader, _, _ = create_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, num_workers=2)
        images, _ = next(iter(train_loader))
    else:
        print("[INFO] Không tìm thấy dữ liệu Train. Đang tạo Tensor giả lập (Dummy Data)...")
        images = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)

    print("\n--- THEO DÕI SỰ BIẾN ĐỔI TENSOR ---")
    print(f"1. Ảnh ban đầu (Images):             {list(images.shape)}")
    
    # Khởi tạo mô hình
    model = OptimizedViT(patch_size=PATCH_SIZE, embedding_dim=EMBEDDING_DIM, img_size=IMAGE_SIZE)
    
    # Chạy mô hình
    output = model(images)
    
    # Tính toán kỳ vọng
    num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2
    expected_tokens = num_patches + 1 # +1 cho [CLS] token
    
    print(f"2. Sau khi Patch + CLS + PosEncode:  {list(output.shape)}")
    
    print("\n" + "-"*50)
    print("--- KIỂM TRA TOÁN HỌC ---")
    print(f"▪️ Số lượng patch (N) = (224/16)^2 = 196 patches")
    print(f"▪️ Kèm thêm 1 [CLS] token -> Tổng cộng = 197 tokens")
    print(f"▪️ Chiều dài vector đặc trưng (Embedding Dim) = {EMBEDDING_DIM}")
    print(f"👉 Shape đúng bắt buộc phải là: [{BATCH_SIZE}, {expected_tokens}, {EMBEDDING_DIM}]")
    
    if list(output.shape) == [BATCH_SIZE, expected_tokens, EMBEDDING_DIM]:
        print("\n✅ PASSED! Kích thước hoàn hảo. Dữ liệu đã sẵn sàng để đi vào lớp Attention!")
    else:
        print("\n❌ FAILED! Có lỗi sai kích thước ở đâu đó.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()