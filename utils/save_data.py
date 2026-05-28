import mne
import os
from pathlib import Path


def save_epochs(epochs, name, folder):
    """
    Saves the combined epochs to a .fif file.
    """
    dir = Path(os.getcwd() , "processed", folder)
    os.makedirs(dir, exist_ok=True)

    save_path = dir / f"{name}.fif"
    print(f"Saving combined epochs to {save_path}...")
    epochs.save(save_path, overwrite=True)