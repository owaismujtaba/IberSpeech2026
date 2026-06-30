import mne
import numpy as np

from mne_bids import BIDSPath, read_raw_bids
from src.utils.logger import get_logger

log = get_logger()


def load_subject_data(bids_root, subject, task, datatype="eeg"):
    """
    Loads raw BIDS data for a specific subject.
    Returns a list of mne.io.Raw objects (one per session/run).
    """
    log.info(f"[BIDS] Loading subject={subject}  task={task}  datatype={datatype}  root={bids_root}")

    bids_path = BIDSPath(root=bids_root, subject=subject, task=task, datatype=datatype)
    match_paths = bids_path.match()
    log.info(f"[BIDS] Found {len(match_paths)} matching BIDS path(s)")

    raws = []
    for i, path in enumerate(match_paths):
        if path.extension in ['.tsv', '.json']:
            log.debug(f"[BIDS]   Skipping non-EEG file: {path.basename}")
            continue
        try:
            log.info(f"[BIDS]   Reading file {i+1}: {path.basename}")
            raw = read_raw_bids(bids_path=path, verbose=False)
            raw.load_data()
            info = raw.info
            duration_s = raw.times[-1]
            log.info(
                f"[BIDS]   Loaded  →  channels={len(info['ch_names'])}  "
                f"sfreq={info['sfreq']:.1f} Hz  duration={duration_s:.1f}s  "
                f"shape={raw.get_data().shape}"
            )
            # Attach the BIDS session entity so callers can use it for naming
            session_id = path.session if path.session else str(i + 1)
            raws.append((session_id, raw))
        except Exception as e:
            log.error(f"[BIDS]   Failed to load {path}: {e}")

    log.info(f"[BIDS] Subject {subject}: {len(raws)} raw object(s) loaded")
    return raws


# Speech-mode prefix in the annotation string → human-readable mode label.
# Per dataset convention: "Real" annotations are Covert, "Silent" are Overt.
_MODE_MAP = {"Real": "Covert", "Silent": "Overt"}


def _events_for_annotations(raw, matched):
    """
    Given a dict ``{annotation_string: synthetic_id}``, return the MNE events array
    and the list of annotation strings aligned to each event row.

    A unique synthetic id per annotation string avoids hash collisions; we rely on
    epoch ``metadata`` (not ``event_id``) to carry the real labels downstream.
    """
    if not matched:
        return np.empty((0, 3), dtype=int), []

    events, id_mapped = mne.events_from_annotations(raw, event_id=matched, verbose=False)
    id_to_string = {v: k for k, v in id_mapped.items()}
    strings = [id_to_string[row_id] for row_id in events[:, 2]]
    return events, strings


def extract_word_speech_events(raw):
    """
    Extract speech-onset events for the Words experiment (Overt + Covert).

    Returns ``(events, meta)`` where ``meta`` is a list of dicts (one per event):
    ``{"mode": "Overt"/"Covert", "word": <normalised label>}``.
    """
    _, event_dict = mne.events_from_annotations(raw, verbose=False)

    matched = {}   # annotation string -> synthetic id
    for idx, ev in enumerate(
        s for s in event_dict
        if "StartSpeech" in s and "WordsExperiment" in s and "Practice" not in s
    ):
        matched[ev] = idx + 1

    events, strings = _events_for_annotations(raw, matched)

    meta = []
    for s in strings:
        mode_prefix = "Real" if s.startswith("Real") else "Silent"
        meta.append({"mode": _MODE_MAP[mode_prefix], "word": s.split("_")[-1].strip().lower()})

    log.info(f"[EVENTS] Words speech  →  {len(events)} event(s)")
    return events, meta


def extract_rest_events(raw):
    """
    Extract non-speech ("Rest") onsets from the Words-experiment fixation periods.

    Returns ``(events, meta)`` with ``meta`` dicts ``{"mode": "Rest", "word": "rest"}``.
    """
    _, event_dict = mne.events_from_annotations(raw, verbose=False)

    matched = {}
    for idx, ev in enumerate(
        s for s in event_dict
        if "WordsExperimentStartFixation" in s and "Practice" not in s
    ):
        matched[ev] = idx + 1

    events, strings = _events_for_annotations(raw, matched)
    meta = [{"mode": "Rest", "word": "rest"} for _ in strings]

    log.info(f"[EVENTS] Words rest (fixation)  →  {len(events)} event(s)")
    return events, meta
