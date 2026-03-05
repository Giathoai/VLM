import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from PIL import Image
from models.vit import VIT
from models.vlm import SeeMoreVLM
from dataloaders.transforms import get_transform

def generate_text(model, image_tensor, prompt_text, tokenizer, device, max_length=50):
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    image_tensor = image_tensor.to(device)
    
    generated_ids = input_ids.clone()
    
    with torch.inference_mode():
        for _ in range(max_length):
            logits = model(image_tensor, generated_ids)
            next_token_logits = logits[:, -1, :]
            
            top_k = 50
            values, indices = torch.topk(next_token_logits, top_k)
            probs = F.softmax(values, dim=-1)
            next_token_index = torch.multinomial(probs, num_samples=1)
            next_token = torch.gather(indices, -1, next_token_index)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    VISION_DIM = 32

    vision_encoder = VIT(embedding_dim=512, num_classes=1, num_layers=6)
    model = SeeMoreVLM(vision_encoder=vision_encoder)
    
    model.load_state_dict(torch.load("weights/seemore_vlm_best.pth", map_location=device))
    model.to(device)
    model.eval()

    dataset = load_dataset("parquet", data_files="data/train.parquet", split="train")
    transform = get_transform(image_size=224, is_train=False)

    OUT_DIR = "eval_outputs"
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n[INFO] Ảnh đánh giá sẽ được lưu tại thư mục: {OUT_DIR}")
    print("="*50)
    
    for i in range(min(10, len(dataset))):
        item = dataset[i]
        
        img_data = item.get('image')
        if img_data is None:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        else:
            image = img_data.convert('RGB')
            
        img_path = os.path.join(OUT_DIR, f"sample_{i+1}.jpg")
        image.save(img_path)
            
        image_tensor = transform(image).unsqueeze(0)
        
        instruction = item.get('instruction', '').strip()
        inputs = item.get('inputs', '').strip()
        prompt = f"USER: {instruction}\n{inputs}\nASSISTANT:".strip()
        
        output_text = generate_text(model, image_tensor, prompt, tokenizer, device)
        
        print(f"--- MẪU THỬ {i+1} ---")
        print(f"Đã lưu ảnh tại: {img_path}")
        print(f"Câu hỏi:\n{prompt}")
        print(f"Mô hình đáp:\n{output_text}")
        print("="*50)

if __name__ == "__main__":
    main()