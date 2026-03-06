import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
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
                
    return tokenizer.decode(generated_ids[0][prompt_length:], skip_special_tokens=True).strip()

def evaluate_with_qwen(judge_model, judge_tokenizer, question, ground_truth, prediction, device):
    prompt = f"""You are a strict evaluator. Compare the Model Prediction with the Ground Truth to see if it correctly answers the Question.
Question: {question}
Ground Truth: {ground_truth}
Model Prediction: {prediction}

Does the Model Prediction correctly convey the same meaning as the Ground Truth? Answer strictly with "Yes" or "No"."""
    
    messages = [
        {"role": "system", "content": "You are a helpful and impartial judge."},
        {"role": "user", "content": prompt}
    ]
    
    text = judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = judge_tokenizer([text], return_tensors="pt").to(device)
    
    with torch.inference_mode():
        generated_ids = judge_model.generate(**model_inputs, max_new_tokens=10, temperature=0.1)
        
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = judge_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    vision_encoder = VIT(embedding_dim=512, num_classes=6, num_layers=6)
    model = SeeMoreVLM(vision_encoder=vision_encoder)
    
    model.load_state_dict(torch.load("weights/seemore_vlm_best (2).pth", map_location=device))
    model.to(device)
    model.eval()

    print("\n[INFO] Đang tải mô hình Giám khảo Qwen-0.5B...")
    judge_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    judge_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", 
        torch_dtype=torch.bfloat16
    ).to(device)
    judge_model.eval()

    dataset = load_dataset("parquet", data_files="data/train-00000-of-00153-038abcb44327f394.parquet", split="train")
    transform = get_transform(image_size=224, is_train=False)

    OUT_DIR = "eval_outputs"
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n[INFO] Ảnh đánh giá sẽ được lưu tại thư mục: {OUT_DIR}")
    print("="*50)
    
    correct_count = 0
    total_evaluated = 0
    
    for i in range(min(100, len(dataset))):
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
        ground_truth = item.get('outputs', '').strip() 
        
        full_question = f"{instruction}\n{inputs}".strip()
        prompt = f"USER: {full_question}\nASSISTANT:".strip()
        
        prediction = generate_text(model, image_tensor, prompt, tokenizer, device)
        
        evaluation = evaluate_with_qwen(judge_model, judge_tokenizer, full_question, ground_truth, prediction, device)
        
        if "yes" in evaluation.lower():
            correct_count += 1
        total_evaluated += 1
        
        print(f"--- MẪU THỬ {i+1} ---")
        print(f"Câu hỏi:\n{full_question}")
        print("-" * 20)
        print(f"Đáp án thật:\n{ground_truth}")
        print("-" * 20)
        print(f"Mô hình đáp:\n{prediction}")
        print("-" * 20)
        print(f"Qwen Chấm Điểm: {evaluation}")
        print("="*50)

    if total_evaluated > 0:
        accuracy = (correct_count / total_evaluated) * 100
        print(f"\n[TỔNG KẾT ĐÁNH GIÁ]")
        print(f"Tổng số mẫu đã test : {total_evaluated}")
        print(f"Số mẫu khớp (Yes)   : {correct_count}")
        print(f"Độ chính xác        : {accuracy:.2f}%")

if __name__ == "__main__":
    main()