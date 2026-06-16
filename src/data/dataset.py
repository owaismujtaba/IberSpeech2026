import mne
import os
from pathlib import Path
import torch
import json
from collections import Counter
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pdb
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*does not conform to MNE naming conventions.*")




def load_and_split_per_file(test_size=0.2, random_state=42, directory='syllable'):
    dir = Path(os.getcwd(), "processed", directory)

    if not dir.exists():
        raise ValueError(f"Directory {dir} does not exist. Please run create_epochs first.")

    epoch_files = list(dir.glob("*.fif"))

    if len(epoch_files) == 0:
        raise ValueError(f"No .fif files found in {dir}. Please check your processed data.")

    unique_events = set()
    for epoch_file in epoch_files:
        epochs = mne.read_epochs(epoch_file, preload=False, verbose=False)
        current_id_to_label = {v: k for k, v in epochs.event_id.items()}
        # Only add event labels that actually exist in the events array of this subject/run
        present_ids = np.unique(epochs.events[:, 2])
        for p_id in present_ids:
            if p_id in current_id_to_label:
                unique_events.add(current_id_to_label[p_id])

    sorted_events = sorted(unique_events)
    target_event_id = {label: idx for idx, label in enumerate(sorted_events)}

    print("Enforcing unified Event ID mapping:")
    for label, idx in target_event_id.items():
        print(f"  {label} -> {idx}")
    print("-" * 40)

    all_train_splits = []
    all_test_splits = []
    rng = np.random.RandomState(random_state)

    for epoch_file in epoch_files:
        print(f"Processing and splitting {epoch_file.name}...")

        epochs = mne.read_epochs(epoch_file, preload=True, verbose=False)
        epochs.set_annotations(None)

        current_id_to_label = {v: k for k, v in epochs.event_id.items()}
        new_events = epochs.events.copy()

        for i in range(len(new_events)):
            old_id = new_events[i, 2]
            label = current_id_to_label[old_id]
            new_events[i, 2] = target_event_id[label]

        epochs.events = new_events
        epochs.event_id = target_event_id

        labels = epochs.events[:, -1]
        class_counts = Counter(labels)

        if all(count >= 2 for count in class_counts.values()):
            train_idx, test_idx = train_test_split(np.arange(len(epochs)), test_size=test_size, random_state=random_state, stratify=labels)
        else:
            train_idx = []
            test_idx = []

            for cls in np.unique(labels):
                cls_idx = np.where(labels == cls)[0]
                rng.shuffle(cls_idx)

                if len(cls_idx) == 1:
                    train_idx.extend(cls_idx)
                else:
                    n_test = max(1, int(round(len(cls_idx) * test_size)))
                    n_test = min(n_test, len(cls_idx) - 1)
                    test_idx.extend(cls_idx[:n_test])
                    train_idx.extend(cls_idx[n_test:])

            train_idx = np.array(train_idx)
            test_idx = np.array(test_idx)
            rng.shuffle(train_idx)
            rng.shuffle(test_idx)

        train_classes = set(labels[train_idx])
        test_classes = set(labels[test_idx])

        for cls, count in class_counts.items():
            if count >= 2 and (cls not in train_classes or cls not in test_classes):
                raise RuntimeError(f"Class {cls} missing after split in {epoch_file.name}")

        all_train_splits.append(epochs[train_idx])
        all_test_splits.append(epochs[test_idx])

    print("-" * 40)
    print("Concatenating all splits...")

    global_train_epochs = mne.concatenate_epochs(all_train_splits)
    global_test_epochs = mne.concatenate_epochs(all_test_splits)

    return global_train_epochs, global_test_epochs