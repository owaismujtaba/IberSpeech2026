"""
Build per subject-session epochs for the Words experiment.

Each saved ``.fif`` contains a single ``mne.Epochs`` object holding both speech
(Overt + Covert) and Rest epochs, with an attached ``metadata`` DataFrame:

    mode      : "Overt" / "Covert" / "Rest"
    word      : normalised word label (or "rest")
    category  : one of the 6 semantic categories, or NaN (pseudowords / rest)
    subject   : e.g. "01"
    session   : e.g. "1"

Downstream code derives task-specific labels from this metadata, so a single
epoch set serves all three decoding tasks (speech_mode / semantic_category / word).
"""
import numpy as np
import pandas as pd

from src.utils.config_parser import load_config
from src.utils.logger import get_logger
from src.data.bids_loader import (
    load_subject_data,
    extract_word_speech_events,
    extract_rest_events,
)
from src.data.labels import load_word_categories
from src.preprocess.preprocess import preprocess_raw, epoch_data
from src.utils.save_data import save_epochs

log = get_logger()

# Event codes carried in events[:, 2]; the real labels live in epoch metadata.
_MODE_CODE = {"Overt": 1, "Covert": 2, "Rest": 3}
_REST_SEED = 42


def _balance_rest(rest_events, rest_meta, n_keep, seed=_REST_SEED):
    """Subsample rest events down to ``n_keep`` so Rest doesn't dominate."""
    if len(rest_events) <= n_keep:
        return rest_events, rest_meta
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(rest_events), size=n_keep, replace=False))
    return rest_events[idx], [rest_meta[i] for i in idx]


def create_epochs():
    """Create Words-experiment epochs (speech + rest) for every configured subject."""
    config = load_config()
    word_to_cat = load_word_categories()

    pp = config["preprocessing"]
    log.info("Creating Words epochs (speech + rest) for all subjects ...")

    for subject in config["dataset"]["subjects"]:
        log.info(f"[CREATE] Subject {subject}")
        raws = load_subject_data(
            bids_root=config["dataset"]["bids_root"],
            subject=subject,
            task=config["dataset"]["task"],
            datatype=config["dataset"]["datatype"],
        )

        for index, (session_id, raw) in enumerate(raws, start=1):
            log.info(f"[CREATE] Subject {subject}  session ses-{session_id} (run {index})")

            raw = preprocess_raw(
                raw,
                l_freq=pp["l_freq"],
                h_freq=pp["h_freq"],
                notch_freq=pp.get("notch_freq", 50.0),
                resample_freq=pp["resample_freq"],
            )

            speech_events, speech_meta = extract_word_speech_events(raw)
            rest_events, rest_meta = extract_rest_events(raw)

            if len(speech_events) == 0:
                log.warning(f"[CREATE] No word speech events for subject {subject}, run {index} — skipped")
                continue

            n_overt = sum(1 for m in speech_meta if m["mode"] == "Overt")
            rest_events, rest_meta = _balance_rest(rest_events, rest_meta, n_keep=max(n_overt, 1))

            # ── Combine speech + rest, sorted by sample time ──────────────────
            all_events = np.vstack([speech_events, rest_events]) if len(rest_events) else speech_events
            all_meta = speech_meta + rest_meta

            # Recode events[:, 2] to mode code; metadata carries the true labels.
            for i, m in enumerate(all_meta):
                all_events[i, 2] = _MODE_CODE[m["mode"]]

            order = np.argsort(all_events[:, 0], kind="stable")
            all_events = all_events[order]
            all_meta = [all_meta[i] for i in order]

            metadata = pd.DataFrame(all_meta)
            metadata["category"] = metadata["word"].map(word_to_cat)  # NaN for pseudowords/rest
            metadata["subject"] = subject
            metadata["session"] = str(session_id)

            epochs = epoch_data(
                raw,
                all_events,
                event_id=_MODE_CODE,
                tmin=pp["tmin"],
                tmax=pp["tmax"],
                baseline=tuple(pp["baseline"]) if pp["baseline"] else None,
                metadata=metadata,
            )

            counts = epochs.metadata["mode"].value_counts().to_dict()
            log.info(f"[CREATE]   mode counts → {counts}")
            save_epochs(epochs, name=f"sub-{subject}_ses-{session_id}", folder="words")


if __name__ == "__main__":
    create_epochs()
