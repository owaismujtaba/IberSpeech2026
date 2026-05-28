import mne

def preprocess_raw(raw, l_freq=1.0, h_freq=100.0, resample_freq=None):
    """
    Applies basic preprocessing: filtering, re-referencing, and resampling.
    """
    print(f"Preprocessing raw data")
    # Drop inconsistent channels and EOGs
    raw.drop_channels(['EOG1', 'EOG2', 'TP9', 'TP10'], on_missing='ignore')
    
    # Set standard EEG montage so Interpolated models can calculate 3D spatial locations
    raw.set_montage('standard_1020', match_case=False, on_missing='ignore')

    # Bandpass filter
    #raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
    
    # Notch filter for power line noise (assuming 50Hz for Europe/Spain)
    #raw.notch_filter(freqs=50, verbose=False)

    # Re-reference to average
    raw.set_eeg_reference('average', projection=False, verbose=False)


    return raw

def epoch_data(raw, events, event_id, tmin=-0.2, tmax=1.0, baseline=(-0.2, 0.0)):
    """
    Creates epochs from continuous raw data.
    """
    epochs = mne.Epochs(
        raw, 
        events, 
        event_id=event_id, 
        tmin=tmin, 
        tmax=tmax, 
        baseline=baseline, 
        preload=True,
        event_repeated='drop',
        verbose=False
    )
    return epochs
