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
import mne



def create_syllable_epochs():
    args = get_args()
    config = load_config(args.config)
    config = merge_args_with_config(args, config)

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load and Preprocess Data
    print("Loading datasets...")
    all_epochs = []
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
            # Basic preprocessing
            index += 1
            raw = preprocess_raw(
                raw,
                l_freq=config['preprocessing']['l_freq'],
                h_freq=config['preprocessing']['h_freq'],
                resample_freq=config['preprocessing']['resample_freq']
            )
            
            events, event_id = extract_events(raw)
            if len(events) == 0:
                print(f"No events found for subject {subject}.")
                continue
                
            epochs = epoch_data(
                raw, 
                events, 
                event_id,
                tmin=config['preprocessing']['tmin'],
                tmax=config['preprocessing']['tmax'],
                baseline=tuple(config['preprocessing']['baseline']) if config['preprocessing']['baseline'] else None
            )
            #pdb.set_trace()
            save_epochs(epochs, name=f"sub-{subject}_ses-{index}", folder='syllable')
    
if __name__ == "__main__":
    create_syllable_epochs()