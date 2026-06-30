"""
Recompute per-session metrics (adding Cohen's kappa and balanced accuracy) for models
whose result JSONs predate the kappa metric — WITHOUT retraining. We reload each saved
LOSO checkpoint, run inference on the held-out session(s), and rewrite the JSON with the
full metric set. Accuracy/F1 are recomputed too and should match the stored values.

Usage:
    python recompute_kappa.py                  # default: 4 CNN baselines, all 3 tasks
"""
import os
import json
import glob

import numpy as np
import torch
import torch.nn as nn

from src.utils.config_parser import load_config
from src.utils.logger import get_logger
from src.data.dataset import make_loso_folds
from src.engine.models import create_model
from src.engine.trainer import make_loader, evaluate, PHYSICAL_SCALE_MODELS
from main import _build_summary, write_combined_csv

log = get_logger()

# Models whose JSONs lack kappa. EEGMamba is intentionally excluded (its results are
# being ignored pending a corrected re-run).
MODELS = ["EEGNetv4", "ShallowFBCSPNet", "Deep4Net", "EEGConformer"]
TASKS = ["speech_mode", "semantic_category", "word"]

_LOG_SOFTMAX_MODELS = {"Deep4Net", "ShallowFBCSPNet", "EEGConformer"}


def recompute_one(config, task, model_name, device):
    config["model"]["name"] = model_name
    norm_scheme = "physical" if model_name in PHYSICAL_SCALE_MODELS else "zscore"
    criterion = nn.NLLLoss() if model_name in _LOG_SOFTMAX_MODELS else nn.CrossEntropyLoss()

    json_path = os.path.join("results", task, f"{model_name}_loso.json")
    old = json.load(open(json_path)) if os.path.exists(json_path) else {}
    old_sessions = old.get("per_session", {})

    session_results = {}
    n_folds = 0
    for held_out, (_, _), val_dict, n_classes, vocab, ch_names in make_loso_folds(task):
        ckpt_path = os.path.join("saved_models", task, model_name, f"sub-{held_out}.pt")
        if not os.path.exists(ckpt_path):
            log.warning(f"[KAPPA] missing checkpoint {ckpt_path} — skipping fold")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        n_chans = ckpt["n_chans"]
        n_times = ckpt["n_times"]
        model = create_model(config, n_chans, n_classes, n_times, ch_names=ch_names)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device).eval()

        for stem, (vd, vl) in val_dict.items():
            loader = make_loader(vd, vl.astype(np.int64), batch_size=config["training"]["batch_size"],
                                 shuffle=False, tag=f"EVAL/{stem}", norm=norm_scheme)
            metrics = evaluate(model, loader, criterion, device)
            session_results[stem] = metrics
            old_acc = old_sessions.get(stem, {}).get("accuracy")
            drift = f"  (was acc={old_acc:.4f})" if old_acc is not None else ""
            log.info(f"[KAPPA] {task}/{model_name}/{stem}  acc={metrics['accuracy']:.4f}  "
                     f"kappa={metrics['kappa']:.4f}{drift}")
        n_folds += 1

    if not session_results:
        log.warning(f"[KAPPA] {task}/{model_name}: no sessions evaluated — JSON left unchanged")
        return None

    summary = _build_summary(task, model_name, session_results)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"[KAPPA] {task}/{model_name}  ({n_folds} folds)  →  "
             f"acc={summary['loso_acc_mean']:.4f}  f1={summary['loso_f1_mean']:.4f}  "
             f"kappa={summary['loso_kappa_mean']:.4f}  → wrote {json_path}")
    return summary


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[KAPPA] device={device}")
    for task in TASKS:
        for model_name in MODELS:
            try:
                recompute_one(config, task, model_name, device)
            except Exception:
                log.exception(f"[KAPPA] FAILED  task={task}  model={model_name}")
    write_combined_csv()


if __name__ == "__main__":
    main()
