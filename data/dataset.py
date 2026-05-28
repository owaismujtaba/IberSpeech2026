import torch
from torch.utils.data import Dataset

class EEGEpochsDataset(Dataset):
    def __init__(self, epochs, transform=None):
        """
        PyTorch Dataset for MNE Epochs.
        Args:
            epochs (mne.Epochs): Preprocessed MNE Epochs object.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Get data as a numpy array: shape (n_epochs, n_channels, n_times)
        # Note: Braindecode models expect input shape (batch_size, n_channels, n_times)
        # However, typically braindecode models sometimes expect a 4D input for 2D convolutions (batch, channels, time, 1).
        # We'll just return 3D (batch, channels, time) and reshape in the training loop or model if needed.
        self.data = epochs.get_data(copy=True)
        self.labels = epochs.events[:, 2] # Event ID is in the 3rd column
        self.transform = transform

        # Map labels to 0-indexed contiguous integers for PyTorch CrossEntropyLoss
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

        # Convert to torch tensors
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return sample, label
