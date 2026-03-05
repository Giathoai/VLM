import os
import urllib.request
import torch
from transformers import AutoTokenizer
from models.vit import VIT
from models.vlm import SeeMoreVLM
from dataloaders.dataset import create_dataloader
from utils import engine
from utils.helpers import set_seeds

def download_hf_parquet(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    file_name = "train-00000-of-00153-038abcb44327f394.parquet"
    file_path = os.path.join(data_dir, file_name)
    url = "https://huggingface.co/datasets/HuggingFaceM4/M3IT/resolve/main/data/train-00000-of-00153-038abcb44327f394.parquet"
    
    if not os.path.exists(file_path):
        print(f"[INFO] Đang tải {file_name} từ HuggingFace (Vui lòng chờ)...")
        urllib.request.urlretrieve(url, file_path)
        print("[INFO] Tải xong!")
    else:
        print(f"[INFO] Đã tìm thấy file {file_name} trong máy.")
        
    return file_path

def main():
    DATA_DIR = "data"
    BATCH_SIZE = 1
    IMAGE_SIZE = 224
    EPOCHS = 10  
    VISION_DIM = 32
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    parquet_path = download_hf_parquet(DATA_DIR)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_dataloader, val_dataloader, test_dataloader = create_dataloader(
        data_path=parquet_path,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_worker=0
    )

    vision_encoder = VIT(embedding_dim=VISION_DIM, num_classes=6, num_layers=6)
    vlm_model = SeeMoreVLM(vision_encoder=vision_encoder, vision_dim=VISION_DIM).to(device)

    optimizer = torch.optim.AdamW(params=vlm_model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    set_seeds(42)

    results = engine.train(model=vlm_model,
                           train_dataloader=train_dataloader,
                           val_dataloader=val_dataloader,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           epochs=EPOCHS,
                           device=device)
    
    torch.save(vlm_model.state_dict(), "weights/seemore_vlm_best.pth")
    print("\n[INFO] Đã lưu mô hình tại weights/seemore_vlm_best.pth")

if __name__ == "__main__":
    main()