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




def load_and_split_per_file(config=None, test_size=0.2, random_state=42, directory='syllable'):
    if config is None:
        from src.utils.config_parser import load_config
        config = load_config()

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

    split_strategy = config.get('dataset', {}).get('split_strategy', 'stratified_holdout')
    
    all_train_splits = []
    all_test_splits = []

    if split_strategy == 'leave_one_subject_out':
        validation_subject = config.get('dataset', {}).get('validation_subject', '14')
        if isinstance(validation_subject, int):
            validation_subject = f"{validation_subject:02d}"
        elif isinstance(validation_subject, str) and len(validation_subject) == 1:
            validation_subject = f"0{validation_subject}"
            
        print(f"Applying Leave-One-Subject-Out strategy (Validation subject: sub-{validation_subject})")
        
        train_files = []
        val_files = []
        for epoch_file in epoch_files:
            if epoch_file.name.startswith(f"sub-{validation_subject}_"):
                val_files.append(epoch_file)
            else:
                train_files.append(epoch_file)
                
        if len(val_files) == 0:
            raise ValueError(f"No files found for validation subject: sub-{validation_subject} in {dir}")
        if len(train_files) == 0:
            raise ValueError(f"No files found for training subjects in {dir}")
            
        for epoch_file in train_files:
            print(f"Processing training subject file {epoch_file.name}...")
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
            all_train_splits.append(epochs)
            
        for epoch_file in val_files:
            print(f"Processing validation subject file {epoch_file.name}...")
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
            all_test_splits.append(epochs)
    else:
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
    
    # Log any missing classes in train set
    train_labels = global_train_epochs.events[:, -1]
    val_labels = global_test_epochs.events[:, -1]
    train_classes = set(train_labels)
    val_classes = set(val_labels)
    missing_in_train = val_classes - train_classes
    if missing_in_train:
        print(f"Warning: The following classes are in the validation set but not in the training set: {missing_in_train}")

    return global_train_epochs, global_test_epochs