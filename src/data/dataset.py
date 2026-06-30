"""
Load Words-experiment epochs and build Leave-One-Subject-Out (LOSO) folds.

A single epoch set (created by ``create_epochs``) carries metadata for all three
decoding tasks; the task selects which epochs to keep and which column to use as the
label. Each task has a *fixed* vocabulary so the model's output dimension is identical
across every LOSO fold.
"""
import os
from pathlib import Path

import mne
import numpy as np

from src.data.labels import CATEGORIES, load_word_categories
from src.utils.config_parser import load_config
from src.utils.logger import get_logger

log = get_logger()

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*does not conform to MNE naming conventions.*")

TASKS = ("speech_mode", "semantic_category", "word")


def task_vocabulary(task: str) -> list:
    """Fixed, sorted label vocabulary for a task → stable label→index mapping."""
    if task == "speech_mode":
        return ["Covert", "Overt", "Rest"]
    if task == "semantic_category":
        return list(CATEGORIES)
    if task == "word":
        return sorted(load_word_categories().keys())
    raise ValueError(f"Unknown task '{task}'. Options: {TASKS}")


def select_labels(metadata, task: str):
    """
    Given an epochs ``metadata`` DataFrame, return ``(mask, labels)`` where ``mask`` is
    a boolean array selecting the epochs used by ``task`` and ``labels`` is the matching
    array of string labels for the kept epochs.
    """
    if task == "speech_mode":
        mask = metadata["mode"].notna().to_numpy()
        labels = metadata.loc[mask, "mode"].to_numpy()
    elif task == "semantic_category":
        mask = metadata["category"].notna().to_numpy()
        labels = metadata.loc[mask, "category"].to_numpy()
    elif task == "word":
        # Real words only == those with a semantic category.
        mask = metadata["category"].notna().to_numpy()
        labels = metadata.loc[mask, "word"].to_numpy()
    else:
        raise ValueError(f"Unknown task '{task}'. Options: {TASKS}")
    return mask, labels


def _subject_of(stem: str) -> str:
    """'sub-03_ses-1' → '03'."""
    return stem.split("_")[0].replace("sub-", "")


def _load_task_data(task: str, directory: str):
    """
    Load every configured subject's epoch file once and return a list of
    ``{"subject", "stem", "X", "y"}`` dicts (labels already mapped to task indices).
    """
    dir_path = Path(os.getcwd(), "processed", directory)
    if not dir_path.exists():
        raise ValueError(f"[DATASET] {dir_path} does not exist. Run create_epochs first.")

    config = load_config()
    included = set(config["dataset"]["subjects"])

    vocab = task_vocabulary(task)
    label_to_idx = {lbl: i for i, lbl in enumerate(vocab)}

    files = sorted(
        f for f in dir_path.glob("*.fif")
        if any(f.stem.startswith(f"sub-{s}") for s in included)
    )
    log.info(f"[DATASET] task='{task}'  {len(files)} file(s)  {len(vocab)} classes")

    records = []
    ch_names = None
    for f in files:
        epochs = mne.read_epochs(f, preload=True, verbose=False)
        if ch_names is None:
            ch_names = epochs.ch_names
        if epochs.metadata is None:
            raise ValueError(
                f"[DATASET] {f.name} has no metadata — it predates the current pipeline. "
                f"Delete processed/words/ and regenerate with workflow.create_words_epochs=true."
            )
        mask, labels = select_labels(epochs.metadata, task)
        if not mask.any():
            log.warning(f"[DATASET] {f.stem}: no epochs for task '{task}' — skipped")
            continue

        X = epochs.get_data()[mask].astype(np.float32)
        y = np.array([label_to_idx[l] for l in labels], dtype=np.int64)
        records.append({"subject": _subject_of(f.stem), "stem": f.stem, "X": X, "y": y})
        log.info(f"[DATASET]   {f.stem}: X={X.shape}  classes_present={len(np.unique(y))}")

    return records, len(vocab), vocab, ch_names


def make_loso_folds(task: str, directory: str = "words"):
    """
    Yield one LOSO fold per subject:

        (held_out_subject, (X_train, y_train), val_dict, n_classes, vocab, ch_names)

    where ``val_dict`` maps each held-out session stem → ``(X, y)``. Training data
    pools every *other* subject's epochs.
    """
    records, n_classes, vocab, ch_names = _load_task_data(task, directory)
    subjects = sorted({r["subject"] for r in records})
    log.info(f"[DATASET] LOSO over {len(subjects)} subject(s): {subjects}")

    for held_out in subjects:
        train_recs = [r for r in records if r["subject"] != held_out]
        val_recs = [r for r in records if r["subject"] == held_out]

        X_train = np.concatenate([r["X"] for r in train_recs])
        y_train = np.concatenate([r["y"] for r in train_recs])
        val_dict = {r["stem"]: (r["X"], r["y"]) for r in val_recs}

        log.info(
            f"[DATASET] Fold held_out=sub-{held_out}  "
            f"train={X_train.shape[0]} epochs from {len(train_recs)} file(s)  "
            f"val={sum(len(y) for _, y in val_dict.values())} epochs from {len(val_dict)} file(s)"
        )
        yield held_out, (X_train, y_train), val_dict, n_classes, vocab, ch_names
