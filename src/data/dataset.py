import mne
import os
from pathlib import Path
import torch
import json
from torch.utils.data import Dataset

    


def load_syllable_epochs(config):
    """
    Loads all syllable epochs from the processed folder and concatenates them.
    Returns a single MNE Epochs object containing all data.
    """

    dir = Path(os.getcwd(), "processed", "syllable")
    if not dir.exists():
        raise ValueError(f"Directory {dir} does not exist. Please run create_syllable_epochs first.")

    epoch_files = list(dir.glob("*.fif"))
    if len(epoch_files) == 0:
        raise ValueError(f"No .fif files found in {dir}. Please check your processed data.")

    all_epochs = []
    for epoch_file in epoch_files:
        print(f"Loading {epoch_file}...")
        epochs = mne.read_epochs(epoch_file, verbose=False)
        all_epochs.append(epochs)

    return all_epochs



def load_words_epochs(config):
    """
    Loads all words epochs from the processed folder and concatenates them.
    Returns a single MNE Epochs object containing all data.
    """
    dir = Path(os.getcwd(), "processed", "words")
    if not dir.exists():
        raise ValueError(f"Directory {dir} does not exist. Please run create_words_epochs first.")

    epoch_files = list(dir.glob("*.fif"))
    if len(epoch_files) == 0:
        raise ValueError(f"No .fif files found in {dir}. Please check your processed data.")

    all_epochs = []
    for epoch_file in epoch_files:
        print(f"Loading {epoch_file}...")
        epochs = mne.read_epochs(epoch_file, verbose=False)
        all_epochs.append(epochs)

    return all_epochs

class EEGEpochsDataset(Dataset):
    def __init__(self, epochs, transform=None):
        """
        PyTorch Dataset for MNE Epochs.
        Args:
            epochs (mne.Epochs): Preprocessed MNE Epochs object.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.data = epochs.get_data(copy=True)
        self.labels = epochs.events[:, 2] 
        self.transform = transform

        unique_labels = sorted(list(set(self.labels)))
        self.label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        self.labels = [self.label_map[lbl] for lbl in self.labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return sample, label
    

    def load_label_map(path='syllable/label_map.json'):
        with open(path, 'r') as f:
            label_map = {int(k): v for k, v in json.load(f).items()}