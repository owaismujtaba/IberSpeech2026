import mne
import hashlib

from mne_bids import BIDSPath, read_raw_bids
import pdb

def load_subject_data(bids_root, subject, task, datatype="eeg"):
    """
    Loads raw BIDS data for a specific subject.
    Returns a list of mne.io.Raw objects (one for each run/session).
    """
    bids_path = BIDSPath(root=bids_root, subject=subject, task=task, datatype=datatype)
    match_paths = bids_path.match()
    
    raws = []
    for path in match_paths:
        if path.extension in ['.tsv', '.json']:
            continue
        try:
            raw = read_raw_bids(bids_path=path, verbose=False)
            raw.load_data()
            raws.append(raw)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            
    return raws


def extract_events(raw, type='Syllables'):
    """
    Extracts Speech events and event_id from a raw object, ignoring 
    modality, experiment type, or whether it is real/silent.
    Maps everything directly to the target syllable.
    """
    # First, get all available annotation strings
    _, event_dict = mne.events_from_annotations(raw, verbose=False)
    
    print('Creating Events, : ', type)
    custom_event_id = {}
    for event_string in event_dict.keys():
        if "StartSpeech" in event_string and f"{type}Experiment" in event_string:
            syllable = event_string.split('_')[-1]
            
            hash_id = int(hashlib.md5(syllable.encode()).hexdigest(), 16) % 10000
            custom_event_id[event_string] = hash_id
    

   
    events, event_id_mapped = mne.events_from_annotations(raw, event_id=custom_event_id, verbose=False)
    
    clean_event_id = {}
    for event_string, h_id in event_id_mapped.items():
        syllable = event_string.split('_')[-1]
        clean_event_id[syllable] = h_id
    
    return events, clean_event_id
