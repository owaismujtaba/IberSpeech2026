import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from .metrics import compute_metrics
from src.engine.models import create_model

def make_loader(x, y, batch_size=64, shuffle=True):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train(config, train_epochs, val_epochs):
    train_data = train_epochs.get_data()
    train_labels = train_epochs.events[:, 2]

    val_data = val_epochs.get_data()
    val_labels = val_epochs.events[:, 2]

    # Ensure train and validation labels are zero-indexed and sequential from 0
    unique_train = np.unique(train_labels)
    unique_val = np.unique(val_labels)
    print("Original unique train labels (first 10):", unique_train[:10], f"... (total {len(unique_train)})")
    print("Original unique validation labels (first 10):", unique_val[:10], f"... (total {len(unique_val)})")
    
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_train)}
    for lbl in unique_val:
        if lbl not in label_to_idx:
            label_to_idx[lbl] = len(label_to_idx)
            
    train_labels = np.array([label_to_idx[lbl] for lbl in train_labels])
    val_labels = np.array([label_to_idx[lbl] for lbl in val_labels])

    n_classes = len(label_to_idx)
    n_chans = train_data.shape[1]
    n_times = train_data.shape[2]

    print("Mapped unique train labels (first 10):", np.unique(train_labels)[:10])
    print("Mapped unique validation labels (first 10):", np.unique(val_labels)[:10])
    print("Train data shape:", train_data.shape)
    print("Validation data shape:", val_data.shape)
    print("Number of classes:", n_classes)

    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    lr = config['training']['lr']
    weight_decay = config['training']['weight_decay']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(config, n_chans, n_classes, n_times)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    criterion = nn.CrossEntropyLoss()
    
    train_loader = make_loader(
        train_data,
        train_labels,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = make_loader(
        val_data,
        val_labels,
        batch_size=batch_size,
        shuffle=False,
    )

    model_name = config['model']['name']
    decoding_type = config['decoding']['type']
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/{model_name}_{decoding_type}.log"
    
    with open(log_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Decoding Type: {decoding_type}\n")
        f.write(f"Number of Channels: {n_chans}\n")
        f.write(f"Number of Classes: {n_classes}\n")
        f.write(f"Timepoints: {n_times}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Weight Decay: {weight_decay}\n")
        f.write("-" * 50 + "\n")

    history = {"train": [], "val": []}
    best_val_acc = -float("inf")
    best_state = None

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_metrics = evaluate(
            model, val_loader, criterion, device
        )

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        val_acc = val_metrics.get("accuracy", 0.0)

        log_line_train = f"Train loss: {train_metrics['loss']:.4f} | Train acc: {train_metrics.get('accuracy', 0.0):.4f} | Train F1: {train_metrics.get('f1_score', 0.0):.4f}"
        log_line_val = f"Val loss:   {val_metrics['loss']:.4f} | Val acc:   {val_acc:.4f} | Val F1: {val_metrics.get('f1_score', 0.0):.4f}"
        
        print(log_line_train)
        print(log_line_val)

        with open(log_path, "a") as f:
            f.write(f"\nEpoch {epoch + 1}/{epochs}\n")
            f.write(log_line_train + "\n")
            f.write(log_line_val + "\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    with open(log_path, "a") as f:
        f.write("-" * 50 + "\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")

    print(f"\nSaved training log to: {log_path}")
    return model, history


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

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