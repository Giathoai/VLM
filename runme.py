import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from PIL import Image
from models.vit import VIT
from models.vlm import SeeMoreVLM
from dataloaders.transforms import get_transform

def generate_text(model, image_tensor, prompt_text, tokenizer, device, max_length=50):
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
    image_tensor = image_tensor.to(device)
    
    generated_ids = input_ids.clone()
    prompt_length = input_ids.shape[1]
    
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
                
    # Cắt bỏ phần câu hỏi, chỉ lấy câu trả lời
    return tokenizer.decode(generated_ids[0][prompt_length:], skip_special_tokens=True).strip()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 1. Khởi tạo Tokenizer và Model
    print("[INFO] Đang tải mô hình...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # LƯU Ý: Hãy đảm bảo các thông số này khớp với file train của bạn
    vision_encoder = VIT(embedding_dim=512, num_classes=6, num_layers=6)
    model = SeeMoreVLM(vision_encoder=vision_encoder)
    
    # Đổi tên file weights nếu cần
    weight_path = "weights/seemore_vlm_best (2).pth"
    if not os.path.exists(weight_path):
        print(f"[LỖI] Không tìm thấy file trọng số tại: {weight_path}")
        return

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    print("[INFO] Tải mô hình thành công!\n")

    transform = get_transform(image_size=224, is_train=False)

    # 2. Vòng lặp tương tác với người dùng
    print("="*50)
    print("CHƯƠNG TRÌNH HỎI ĐÁP VỚI ẢNH (Gõ 'q' hoặc 'quit' để thoát)")
    print("="*50)

    while True:
        # Nhập đường dẫn ảnh
        img_path = input("\n🖼️ Nhập đường dẫn ảnh (ví dụ: test.jpg): ").strip()
        if img_path.lower() in ['q', 'quit']:
            break
            
        if not os.path.exists(img_path):
            print("❌ Lỗi: Không tìm thấy ảnh. Vui lòng kiểm tra lại đường dẫn!")
            continue

        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)
        except Exception as e:
            print(f"❌ Lỗi khi đọc ảnh: {e}")
            continue

        # Nhập câu hỏi
        question = input("❓ Nhập câu hỏi của bạn: ").strip()
        if question.lower() in ['q', 'quit']:
            break

        # Tạo prompt đúng định dạng
        prompt = f"USER: {question}\nASSISTANT:".strip()
        
        # Sinh câu trả lời
        print("🤖 Mô hình đang suy nghĩ...")
        answer = generate_text(model, image_tensor, prompt, tokenizer, device)
        
        print("-" * 30)
        print(f"💡 Trả lời: {answer}")
        print("-" * 30)

    print("\n👋 Đã thoát chương trình. Hẹn gặp lại!")

if __name__ == "__main__":
    main()