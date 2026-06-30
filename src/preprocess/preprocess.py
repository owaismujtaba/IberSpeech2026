import mne
import numpy as np
from src.utils.logger import get_logger

log = get_logger()


def preprocess_raw(raw, l_freq=0.1, h_freq=75.0, notch_freq=50.0, resample_freq=None):
    """
    Full preprocessing pipeline:
      1. Bandpass filter (l_freq – h_freq Hz)
      2. Notch filter at notch_freq Hz
      3. ICA (auto-remove ocular + cardiac components)
      4. Drop non-EEG channels (EOG kept until after ICA for artifact detection)
      5. Average reference
    Data is kept in Volts (MNE standard). µV values are logged for readability.
    """
    info = raw.info
    log.info(
        f"[PREPROC] Input  →  channels={len(info['ch_names'])}  "
        f"sfreq={info['sfreq']:.1f} Hz  "
        f"amplitude range=[{raw.get_data().min()*1e6:.2f}, {raw.get_data().max()*1e6:.2f}] µV"
    )

    raw.set_montage('standard_1020', match_case=False, on_missing='ignore')

    # 1. Bandpass filter
    log.info(f"[PREPROC] Bandpass  {l_freq}–{h_freq} Hz")
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)

    # 3. Notch filter
    log.info(f"[PREPROC] Notch filter at {notch_freq} Hz")
    raw.notch_filter(freqs=notch_freq, verbose=False)

    # 4. ICA for artifact removal
    log.info("[PREPROC] Fitting ICA to remove eye/muscle artifacts ...")
    try:
        present_ref = [ch for ch in ['EOG1', 'EOG2'] if ch in raw.ch_names]
        if not present_ref:
            present_ref = [ch for ch in ['TP9', 'TP10'] if ch in raw.ch_names]

        ica = mne.preprocessing.ICA(n_components=15, max_iter='auto', random_state=97)
        ica.fit(raw, picks='eeg', verbose=False)

        if present_ref:
            log.info(f"[PREPROC] Detecting eye artifacts using reference channels: {present_ref}")
            eog_indices, _ = ica.find_bads_eog(raw, ch_name=present_ref, threshold=3.0, verbose=False)
            ica.exclude = eog_indices
            log.info(f"[PREPROC] Excluded {len(eog_indices)} ICA components: {eog_indices}")
        else:
            log.info("[PREPROC] No reference channels found for automatic component rejection.")

        ica.apply(raw, verbose=False)
    except Exception as e:
        log.warning(f"[PREPROC] ICA fitting failed: {e}. Proceeding without ICA.")

    # 5. Drop non-EEG channels after ICA (EOG channels were needed for artifact detection)
    before = len(raw.ch_names)
    raw.drop_channels(['EOG1', 'EOG2', 'TP9', 'TP10'], on_missing='ignore')
    dropped = before - len(raw.ch_names)
    if dropped:
        log.info(f"[PREPROC] Dropped {dropped} non-EEG channel(s)  →  {len(raw.ch_names)} remaining")

    # 6. Average reference
    raw.set_eeg_reference('average', projection=False, verbose=False)
    log.info("[PREPROC] Applied average reference")

    data = raw.get_data()
    log.info(
        f"[PREPROC] Output  →  shape={data.shape}  "
        f"amplitude range=[{data.min()*1e6:.2f}, {data.max()*1e6:.2f}] µV  "
        f"std={data.std()*1e6:.4f} µV"
    )

    # Optional resampling
    if resample_freq and resample_freq != raw.info['sfreq']:
        log.info(f"[PREPROC] Resampling  {raw.info['sfreq']:.1f} Hz → {resample_freq} Hz")
        raw.resample(resample_freq, verbose=False)
        log.info(f"[PREPROC] After resample  →  shape={raw.get_data().shape}")

    return raw


def epoch_data(raw, events, event_id, tmin=-0.2, tmax=1.5, baseline=(-0.2, 0.0),
               metadata=None):
    """
    Creates epochs from continuous raw data.

    ``metadata`` (optional pandas DataFrame, aligned row-for-row with ``events``)
    is attached to the resulting epochs so callers can derive task-specific labels.
    """
    log.info(
        f"[EPOCH] Creating epochs  tmin={tmin}s  tmax={tmax}s  "
        f"baseline={baseline}  n_events={len(events)}  n_classes={len(event_id)}"
    )

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        metadata=metadata,
        event_repeated='drop',
        verbose=False
    )

    data = epochs.get_data()
    log.info(
        f"[EPOCH] Output  →  shape={data.shape}  "
        f"(n_epochs={data.shape[0]}, n_channels={data.shape[1]}, n_times={data.shape[2]})  "
        f"sfreq={epochs.info['sfreq']:.1f} Hz  "
        f"amplitude range=[{data.min()*1e6:.2f}, {data.max()*1e6:.2f}] µV"
    )

    counts = {k: int((epochs.events[:, 2] == v).sum()) for k, v in epochs.event_id.items()}
    min_c, max_c = min(counts.values()), max(counts.values())
    log.info(f"[EPOCH] Trials per class  min={min_c}  max={max_c}  total={len(epochs)}")

    return epochs
