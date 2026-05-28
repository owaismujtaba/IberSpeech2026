import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from utils.config_parser import load_config, get_args, merge_args_with_config
from data.bids_loader import load_subject_data, extract_events
from data.preprocess import preprocess_raw, epoch_data
from data.dataset import EEGEpochsDataset
from models.model_factory import create_model
from engine.trainer import train_one_epoch, evaluate
from utils.save_data import save_epochs
from data.create_syllable_epochs import create_syllable_epochs
from data.dataset import load_syllable_epochs
import mne
import pdb

def main():
    # 1. Setup configpip
    args = get_args()
    config = load_config(args.config)
    config = merge_args_with_config(args, config)

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if config['workflow']['create_syllable_epochs']:
        create_syllable_epochs()
    else:
        print("Loading syllable epochs...")
        all_epochs = load_syllable_epochs(config)
        combined_epochs = mne.concatenate_epochs(all_epochs)
    
    # Update config based on loaded data
    config['model']['in_chans'] = len(combined_epochs.ch_names)
    config['model']['input_window_samples'] = len(combined_epochs.times)
    # Number of unique event types
    config['model']['n_classes'] = len(set(combined_epochs.events[:, 2]))
    print(f"Data shape: {combined_epochs.get_data().shape}")
    print(f"Number of classes: {config['model']['n_classes']}")

    # 3. Create PyTorch Datasets and DataLoaders
    full_dataset = EEGEpochsDataset(combined_epochs)
    
    # 80-20 Train-Test Split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # 4. Initialize Model, Optimizer, Loss
    print(f"Initializing {config['model']['name']}...")
    model = create_model(
        model_name=config['model']['name'],
        n_classes=config['model']['n_classes'],
        in_chans=config['model']['in_chans'],
        input_window_samples=config['model']['input_window_samples'],
        device=device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['lr'], 
        weight_decay=config['training']['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    print("Starting training...")
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1_score']:.4f}")

        val_metrics = evaluate(model, test_loader, criterion, device)
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    main()
