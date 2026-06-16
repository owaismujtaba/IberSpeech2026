import mne
import os
from pathlib import Path

from src.utils.config_parser import load_config
from src.data.bids_loader import load_subject_data, extract_events
from src.preprocess.preprocess import preprocess_raw, epoch_data
from src.utils.save_data import save_epochs

def create_epochs(create_syllable=True, create_words=True):
    config = load_config()

    if not create_syllable and not create_words:
        print("Neither syllable nor word epoch creation is enabled.")
        return

    print("Loading datasets...")
    for subject in config['dataset']['subjects']:
        print(f"Processing subject {subject}...")
        raws = load_subject_data(
            bids_root=config['dataset']['bids_root'],
            subject=subject,
            task=config['dataset']['task'],
            datatype=config['dataset']['datatype']
        )
        index = 0
        for raw in raws:
            index += 1
            print(f"Preprocessing run {index} for subject {subject}...")
            
            # Preprocess the raw data once
            raw = preprocess_raw(
                raw,
                l_freq=config['preprocessing']['l_freq'],
                h_freq=config['preprocessing']['h_freq'],
                resample_freq=config['preprocessing']['resample_freq']
            )
            
            # 1. Handle syllable epochs
            if create_syllable:
                events, event_id = extract_events(raw, type='Syllables')
                if len(events) == 0:
                    print(f"No syllable events found for subject {subject}, run {index}.")
                else:
                    epochs = epoch_data(
                        raw, 
                        events, 
                        event_id,
                        tmin=config['preprocessing']['tmin'],
                        tmax=config['preprocessing']['tmax'],
                        baseline=tuple(config['preprocessing']['baseline']) if config['preprocessing']['baseline'] else None
                    )
                    save_epochs(epochs, name=f"sub-{subject}_ses-{index}", folder='syllable')

            # 2. Handle word epochs
            if create_words:
                events, event_id = extract_events(raw, type='Words')
                if len(events) == 0:
                    print(f"No word events found for subject {subject}, run {index}.")
                else:
                    epochs = epoch_data(
                        raw, 
                        events, 
                        event_id,
                        tmin=config['preprocessing']['tmin'],
                        tmax=config['preprocessing']['tmax'],
                        baseline=tuple(config['preprocessing']['baseline']) if config['preprocessing']['baseline'] else None
                    )
                    save_epochs(epochs, name=f"sub-{subject}_ses-{index}", folder='words')

if __name__ == "__main__":
    # If run directly, read from config workflow setting
    config = load_config()
    create_epochs(
        create_syllable=config['workflow']['create_syllable_epochs'],
        create_words=config['workflow']['create_words_epochs']
    )
