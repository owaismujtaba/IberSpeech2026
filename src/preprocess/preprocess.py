import mne

def preprocess_raw(raw, l_freq=1.0, h_freq=100.0, resample_freq=None):
    """
    Applies basic preprocessing: filtering, ICA, re-referencing, resampling, and scaling.
    """
    print(f"Preprocessing raw data: l_freq={l_freq}, h_freq={h_freq}, resample_freq={resample_freq}")
    
    raw.set_montage('standard_1020', match_case=False, on_missing='ignore')
    
    # 1. Bandpass filter
    print(f"Applying bandpass filter ({l_freq} - {h_freq} Hz)...")
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
    
    # 2. Notch filter (50 Hz)
    print("Applying notch filter (50 Hz)...")
    raw.notch_filter(freqs=50.0, verbose=False)

    # 3. ICA (Independent Component Analysis) for artifact removal
    print("Fitting ICA to remove eye/muscle artifacts...")
    try:
        # Check if EOG channels are present in the raw data
        present_ref = [ch for ch in ['EOG1', 'EOG2'] if ch in raw.ch_names]
        
        # If not present, fall back to TP9 and TP10
        if not present_ref:
            present_ref = [ch for ch in ['TP9', 'TP10'] if ch in raw.ch_names]
        
        # Fit ICA on EEG channels
        ica = mne.preprocessing.ICA(n_components=15, max_iter='auto', random_state=97)
        ica.fit(raw, picks='eeg', verbose=False)
        
        if present_ref:
            print(f"Detecting eye artifacts using reference channels: {present_ref}")
            eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=present_ref, threshold=3.0, verbose=False)
            ica.exclude = eog_indices
            print(f"Excluded {len(eog_indices)} ICA components: {eog_indices}")
        else:
            print("No reference channels found for automatic component rejection.")
            
        # Apply ICA
        ica.apply(raw, verbose=False)
    except Exception as e:
        print(f"Warning: ICA fitting failed with error: {e}. Proceeding without ICA.")

    # 4. Drop irrelevant channels (EOG and mastoids)
    print("Dropping irrelevant channels...")
    raw.drop_channels(['EOG1', 'EOG2', 'TP9', 'TP10'], on_missing='ignore')

    print("Setting average reference...")
    raw.set_eeg_reference('average', projection=False, verbose=False)

    if resample_freq is not None and resample_freq != 'None' and resample_freq != 'none':
        print(f"Resampling data to {resample_freq} Hz...")
        raw.resample(sfreq=float(resample_freq), verbose=False)
    print("Scaling EEG data to µV...")
    raw.apply_function(lambda x: x * 1e6, picks='eeg', verbose=False)

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
