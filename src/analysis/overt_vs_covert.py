"""
Overt -> covert degradation analysis on UGR-MINDVOICE, WITHOUT retraining.

The lexical tasks (semantic_category, word) pool each item's overt and covert
productions into a single label, so the headline Table~\\ref{tab:main} numbers mix the
two speaking modes. To quantify how much decoding degrades from overt to covert
(imagined-like) speech, we reuse the *already-trained* LOSO checkpoints (each trained
on the pooled overt+covert epochs of 13 subjects) and simply evaluate every held-out
subject's epochs split by speaking mode: overt-only vs covert-only.

This is inference only — the same model, scored on two disjoint subsets of the held-out
data — so it isolates the overt/covert gap without changing the trained decision
function. Speech-mode (which *is* the overt/covert/rest contrast) is excluded.

Outputs:
  * results/overt_vs_covert/<task>_<model>.json  (per-subject + LOSO means per mode),
  * results/overt_vs_covert/summary.csv,
  * a stdout table and a ready-to-cite LaTeX snippet.

Run (CPU, to match the kappa recompute):
    CUDA_VISIBLE_DEVICES="" python -m src.analysis.overt_vs_covert
"""
import os
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import mne

from src.utils.config_parser import load_config
from src.utils.logger import get_logger
from src.data.dataset import task_vocabulary, select_labels, _subject_of
from src.engine.models import create_model
from src.engine.trainer import make_loader, evaluate, PHYSICAL_SCALE_MODELS

warnings.filterwarnings("ignore")
log = get_logger()

# Only the lexical tasks pool overt+covert; speech_mode IS the overt/covert contrast.
TASKS = ["semantic_category", "word"]

MODELS = [
    ("EEGNetv4",        "EEGNet"),
    ("ShallowFBCSPNet", "ShallowNet"),
    ("Deep4Net",        "Deep4Net"),
    ("EEGConformer",    "Conformer"),
    ("LaBraM",          "LaBraM"),
    ("EEGMamba",        "EEGMamba"),
]

MODES = ["Overt", "Covert"]
_LOG_SOFTMAX_MODELS = {"Deep4Net", "ShallowFBCSPNet", "EEGConformer"}
OUT_DIR = os.path.join("results", "overt_vs_covert")


def load_records_with_mode(task, directory="words"):
    """Like dataset._load_task_data, but also return the per-epoch speaking mode.

    Returns (records, n_classes, vocab, ch_names) where each record is
    {subject, stem, X, y, mode} and `mode` is a string array aligned with X/y.
    """
    config = load_config()
    included = set(config["dataset"]["subjects"])
    vocab = task_vocabulary(task)
    label_to_idx = {lbl: i for i, lbl in enumerate(vocab)}

    dir_path = Path(os.getcwd(), "processed", directory)
    files = sorted(f for f in dir_path.glob("*.fif")
                   if any(f.stem.startswith(f"sub-{s}") for s in included))

    records, ch_names = [], None
    for f in files:
        ep = mne.read_epochs(f, preload=True, verbose=False)
        if ch_names is None:
            ch_names = ep.ch_names
        md = ep.metadata
        mask, labels = select_labels(md, task)
        if not mask.any():
            continue
        X = ep.get_data()[mask].astype(np.float32)
        y = np.array([label_to_idx[l] for l in labels], dtype=np.int64)
        mode = md.loc[mask, "mode"].to_numpy().astype(str)
        records.append({"subject": _subject_of(f.stem), "stem": f.stem,
                        "X": X, "y": y, "mode": mode})
    return records, len(vocab), vocab, ch_names


def _criterion(model_name):
    return nn.NLLLoss() if model_name in _LOG_SOFTMAX_MODELS else nn.CrossEntropyLoss()


def evaluate_one(config, task, model_name, device):
    config["model"]["name"] = model_name
    norm = "physical" if model_name in PHYSICAL_SCALE_MODELS else "zscore"
    criterion = _criterion(model_name)

    records, n_classes, vocab, ch_names = load_records_with_mode(task)
    by_subject = {}
    for r in records:
        by_subject.setdefault(r["subject"], []).append(r)

    # subject -> mode -> list of session accuracies / kappas
    per_subject = {m: {"accuracy": {}, "kappa": {}} for m in MODES}

    for held_out, recs in sorted(by_subject.items()):
        ckpt_path = os.path.join("saved_models", task, model_name, f"sub-{held_out}.pt")
        if not os.path.exists(ckpt_path):
            log.warning(f"[O/C] missing checkpoint {ckpt_path} — skipping subject {held_out}")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = create_model(config, ckpt["n_chans"], n_classes, ckpt["n_times"], ch_names=ch_names)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device).eval()

        for r in recs:  # one or more sessions for this subject
            for mode in MODES:
                sel = r["mode"] == mode
                if not sel.any():
                    continue
                loader = make_loader(r["X"][sel], r["y"][sel].astype(np.int64),
                                     batch_size=config["training"]["batch_size"],
                                     shuffle=False, tag=f"O/C/{r['stem']}/{mode}", norm=norm)
                m = evaluate(model, loader, criterion, device)
                per_subject[mode]["accuracy"].setdefault(held_out, []).append(m["accuracy"])
                per_subject[mode]["kappa"].setdefault(held_out, []).append(m["kappa"])

    # Per-subject score = mean over that subject's sessions; LOSO = mean over subjects.
    summary = {"task": task, "model": model_name, "n_classes": n_classes}
    for mode in MODES:
        acc = {s: float(np.mean(v)) for s, v in per_subject[mode]["accuracy"].items()}
        kap = {s: float(np.mean(v)) for s, v in per_subject[mode]["kappa"].items()}
        a = np.array(list(acc.values())); k = np.array(list(kap.values()))
        summary[mode] = {
            "n_subjects": len(acc),
            "acc_mean": float(a.mean()) if len(a) else 0.0,
            "acc_std":  float(a.std())  if len(a) else 0.0,
            "kappa_mean": float(k.mean()) if len(k) else 0.0,
            "per_subject_acc": acc,
            "per_subject_kappa": kap,
        }
    o, c = summary["Overt"], summary["Covert"]
    summary["acc_drop_overt_minus_covert"] = o["acc_mean"] - c["acc_mean"]

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{task}_{model_name}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[O/C] device={device}")

    rows = []
    for task in TASKS:
        chance = 1.0 / len(task_vocabulary(task))
        print(f"\n=== {task}  (chance = {chance*100:.1f}%) ===")
        print(f"{'Model':<12}{'Overt acc':>11}{'Covert acc':>12}{'drop':>8}"
              f"{'Overt kappa':>13}{'Covert kappa':>14}")
        for mj, label in MODELS:
            try:
                s = evaluate_one(config, task, mj, device)
            except Exception:
                log.exception(f"[O/C] FAILED task={task} model={mj}")
                continue
            o, c = s["Overt"], s["Covert"]
            print(f"{label:<12}{o['acc_mean']*100:>10.1f}%{c['acc_mean']*100:>11.1f}%"
                  f"{s['acc_drop_overt_minus_covert']*100:>+7.1f}%"
                  f"{o['kappa_mean']:>13.3f}{c['kappa_mean']:>14.3f}")
            rows.append({
                "task": task, "model": label, "chance": round(chance, 4),
                "overt_acc": round(o["acc_mean"], 6), "overt_acc_std": round(o["acc_std"], 6),
                "covert_acc": round(c["acc_mean"], 6), "covert_acc_std": round(c["acc_std"], 6),
                "acc_drop": round(s["acc_drop_overt_minus_covert"], 6),
                "overt_kappa": round(o["kappa_mean"], 6),
                "covert_kappa": round(c["kappa_mean"], 6),
            })

    if rows:
        import csv
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n[CSV] wrote {len(rows)} rows → {os.path.join(OUT_DIR, 'summary.csv')}")


if __name__ == "__main__":
    main()
