import torch
from tqdm.auto import tqdm

def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss = 0

    for batch, (images, input_ids, labels) in enumerate(dataloader):
        images = images.to(device)
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits = model(images, input_ids)
        
        logits_text = logits[:, 196:-1, :].contiguous()

        loss = loss_fn(logits_text.reshape(-1, logits_text.size(-1)), labels.reshape(-1))
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / len(dataloader)

def eval_step(model, dataloader, loss_fn, device):
    model.eval()
    eval_loss = 0

    with torch.inference_mode():
        for batch, (images, input_ids, labels) in enumerate(dataloader):
            images = images.to(device)
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(images, input_ids)
            
            logits_text = logits[:, 196:-1, :].contiguous()

            loss = loss_fn(logits_text.reshape(-1, logits_text.size(-1)), labels.reshape(-1))
            eval_loss += loss.item()

    return eval_loss / len(dataloader)

def train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs, device):
    results = {"train_loss": [], "val_loss": []}

    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model, train_dataloader, loss_fn, optimizer, device)
        
        val_loss = 0
        if val_dataloader is not None:
            val_loss = eval_step(model, val_dataloader, loss_fn, device)

        print(f"Epoch: {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)

    return results