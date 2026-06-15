import mne
import hashlib

from src.utils.config_parser import load_config
from src.data.bids_loader import load_subject_data
from src.preprocess.preprocess import preprocess_raw, epoch_data
from src.utils.save_data import save_epochs


def extract_word_events(raw):
    _, event_dict = mne.events_from_annotations(raw, verbose=False)
    
    custom_event_id = {}
    for event_string in event_dict.keys():
        if "StartSpeech" in event_string and "WordsExperiment" in event_string:
            syllable = event_string.split('_')[-1]
            hash_id = int(hashlib.md5(syllable.encode()).hexdigest(), 16) % 10000
            custom_event_id[event_string] = hash_id
    

   
    events, event_id_mapped = mne.events_from_annotations(raw, event_id=custom_event_id, verbose=False)
    
    clean_event_id = {}
    for event_string, h_id in event_id_mapped.items():
        syllable = event_string.split('_')[-1]
        clean_event_id[syllable] = h_id
    
    return events, clean_event_id


def create_words_epochs():
    
    config = load_config()
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
            
            events, event_id = extract_word_events(raw)
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
            
            save_epochs(epochs, name=f"sub-{subject}_ses-{index}", folder='words')
    
if __name__ == "__main__":
    create_words_epochs()
