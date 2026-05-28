import mne
from mne_bids import BIDSPath, read_raw_bids

def load_subject_data(bids_root, subject, task, datatype="eeg"):
    """
    Loads raw BIDS data for a specific subject.
    Returns a list of mne.io.Raw objects (one for each run/session).
    """
    bids_path = BIDSPath(root=bids_root, subject=subject, task=task, datatype=datatype)
    
    # Find all matching files (different sessions/runs)
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

def extract_events(raw):
    """
    Extracts events and event_id from a raw object.
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    return events, event_id
