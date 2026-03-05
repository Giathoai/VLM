import torch
from transformers import AutoTokenizer
from models.vit import VIT
from models.vlm import SeeMoreVLM
from dataloaders.dataset import create_dataloader
from utils import engine
from utils.helpers import set_seeds

def main():
    DATA_DIR = "data"
    BATCH_SIZE = 4
    IMAGE_SIZE = 224
    EPOCHS = 10  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_dataloader, val_dataloader, test_dataloader = create_dataloader(
        data_path="data/train.parquet", 
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_worker=4
    )

    vision_encoder = VIT(embedding_dim=32, num_classes=1, num_layers=2)
    vlm_model = SeeMoreVLM(vision_encoder=vision_encoder).to(device)

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