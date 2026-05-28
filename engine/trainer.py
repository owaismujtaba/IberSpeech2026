import torch
import torch.nn as nn
from tqdm import tqdm
from .metrics import compute_metrics

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)

        # Braindecode models handle 3D inputs natively (batch_size, n_channels, n_times)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        # If output is (batch, classes, ...), squeeze extra dimensions
        if outputs.ndim > 2:
            outputs = outputs.squeeze()
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": total_loss / (batch_idx + 1)})

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    return metrics

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)


            outputs = model(inputs)
            
            if outputs.ndim > 2:
                outputs = outputs.squeeze()
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    return metrics
