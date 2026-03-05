import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from functools import partial
from .transforms import get_transform

NUM_WORKER = 0

class VLMInstructDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, image_size=224, is_train=True):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.transform = get_transform(image_size=image_size, is_train=is_train)
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        img_data = item.get('image')
        if img_data is None:
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
        else:
            image = img_data.convert('RGB')
            
        image_tensor = self.transform(image)
        
        instruction = item.get('instruction', '').strip()
        inputs = item.get('inputs', '').strip()
        outputs = item.get('outputs', '').strip()
        
        prompt = f"{instruction}\n{inputs}".strip()
        full_text = f"USER: {prompt}\nASSISTANT: {outputs}{self.tokenizer.eos_token}"
        
        return image_tensor, full_text

def vlm_collate_fn(batch, tokenizer):
    images = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    
    images_batched = torch.stack(images, dim=0)
    
    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512 
    )
    
    labels = tokenized["input_ids"].clone()
    labels[tokenized["attention_mask"] == 0] = -100 
    
    return images_batched, tokenized["input_ids"], labels

def create_dataloader(data_path: str, tokenizer, batch_size: int, image_size: int, num_worker: int):
    if data_path.endswith('.parquet'):
        full_ds = load_dataset("parquet", data_files=data_path, split="train")
    else:
        full_ds = load_dataset("parquet", data_dir=data_path, split="train")

    split_90_10 = full_ds.train_test_split(test_size=0.1, seed=42)
    test_hf = split_90_10['test']
    train_val_hf = split_90_10['train']

    split_train_val = train_val_hf.train_test_split(test_size=(2/9), seed=42)
    train_hf = split_train_val['train']
    val_hf = split_train_val['test']

    print(f"[INFO] Tỉ lệ dữ liệu - Train: {len(train_hf)} | Val: {len(val_hf)} | Test: {len(test_hf)}")

    train_dataset = VLMInstructDataset(train_hf, tokenizer, image_size, is_train=True)
    val_dataset = VLMInstructDataset(val_hf, tokenizer, image_size, is_train=False)
    test_dataset = VLMInstructDataset(test_hf, tokenizer, image_size, is_train=False)

    collate_fn = partial(vlm_collate_fn, tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKER, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKER, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKER, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader