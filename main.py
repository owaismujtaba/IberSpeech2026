"""
Train EEG decoders on the Words experiment with a Leave-One-Subject-Out (LOSO) split.

Runs every (task × model) combination listed in config.yaml. For each combination a
fresh model is trained per LOSO fold (one held-out subject), and results are aggregated
to a LOSO mean ± std across subjects.

Tasks:
  speech_mode        — Overt vs Covert vs Rest
  semantic_category  — 6 semantic categories
  word               — 60 real Spanish words
"""
import os
import csv
import glob
import json

import matplotlib
matplotlib.use("Agg")  # headless — save figures, never open a window
import numpy as np

from src.utils.config_parser import load_config
from src.utils.logger import get_logger
from src.data.create_epochs import create_epochs
from src.data.dataset import make_loso_folds, TASKS
from src.engine.trainer import train
from src.engine.plot_results import plot_per_subject, plot_subject_vs_session

log = get_logger()

import os
import tempfile

# Force a valid cache directory
cache_dir = os.path.join(os.getcwd(), "torch_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
os.environ["TORCH_HOME"] = cache_dir

# Now import torch
import torch


def _subject_of(stem):
    return stem.split("_")[0].replace("sub-", "")


def _build_summary(task, model_name, session_results):
    """Aggregate per-session results into a LOSO summary dict (mean over subjects)."""
    per_subject = {}  # metric -> {subject -> [session values]}
    metric_keys = ("accuracy", "balanced_accuracy", "f1_score", "kappa")
    for stem, m in session_results.items():
        sub = _subject_of(stem)
        for k in metric_keys:
            if k in m:
                per_subject.setdefault(k, {}).setdefault(sub, []).append(m[k])

    # Per-subject score = mean over that subject's sessions.
    subj = {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in per_subject.items()}

    def _mean(k):
        vals = list(subj.get(k, {}).values())
        return float(np.mean(vals)) if vals else 0.0

    def _std(k):
        vals = list(subj.get(k, {}).values())
        return float(np.std(vals)) if vals else 0.0

    return {
        "task": task,
        "model": model_name,
        "n_subjects": len(subj.get("accuracy", {})),
        "loso_acc_mean": _mean("accuracy"),
        "loso_acc_std": _std("accuracy"),
        "loso_balacc_mean": _mean("balanced_accuracy"),
        "loso_balacc_std": _std("balanced_accuracy"),
        "loso_f1_mean": _mean("f1_score"),
        "loso_kappa_mean": _mean("kappa"),
        "per_subject_acc": subj.get("accuracy", {}),
        "per_session": session_results,
    }


def run_task_model(config, task, model_name):
    """Run all LOSO folds for one (task, model) and return aggregated results."""
    config["model"]["name"] = model_name
    seed = config.get("training", {}).get("seed", 42)
    log.info("#" * 70)
    log.info(f"# TASK={task}  MODEL={model_name}")
    log.info("#" * 70)

    out_dir = os.path.join("results", task)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{model_name}_loso.json")

    session_results = {}   # stem -> metrics (held-out sessions across all folds)
    for held_out, (X_train, y_train), val_dict, n_classes, vocab, ch_names in make_loso_folds(task):
        ckpt_path = os.path.join("saved_models", task, model_name, f"sub-{held_out}.pt")
        _, _, per_subject = train(
            config, X_train, y_train, val_dict, n_classes,
            tag=f"{task}/{model_name}/held_out=sub-{held_out}",
            save_path=ckpt_path, vocab=vocab, seed=seed, ch_names=ch_names,
        )
        session_results.update(per_subject)

        # ── Flush metrics to disk after every fold so partial runs are never lost ──
        summary = _build_summary(task, model_name, session_results)
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        write_combined_csv()
        log.info(f"[RUN] {task}/{model_name}  fold sub-{held_out} done  "
                 f"(running LOSO acc={summary['loso_acc_mean']:.4f} over "
                 f"{summary['n_subjects']} subject(s))")

    # ── Final plots once all folds are in ─────────────────────────────────────
    tag = f"{task} — {model_name} (LOSO)"
    plot_per_subject(session_results, metric="accuracy", title=tag,
                     save_path=os.path.join(out_dir, f"{model_name}_per_session.png"))
    plot_subject_vs_session(session_results, metric="accuracy", title=tag,
                            save_path=os.path.join(out_dir, f"{model_name}_subject_vs_session.png"))

    log.info(f"[RUN] {task}/{model_name}  LOSO acc = "
             f"{summary['loso_acc_mean']:.4f} ± {summary['loso_acc_std']:.4f}")
    return summary


def main():
    config = load_config()

    if config["workflow"].get("create_words_epochs", False):
        create_epochs()

    tasks = config["decoding"].get("tasks", list(TASKS))
    models = config.get("models", [config.get("model", {}).get("name", "EEGNetv4")])

    all_summaries = []
    for task in tasks:
        for model_name in models:
            try:
                all_summaries.append(run_task_model(config, task, model_name))
            except Exception:
                log.exception(f"[RUN] FAILED  task={task}  model={model_name}")

    # ── Final summary table ───────────────────────────────────────────────────
    log.info("=" * 70)
    log.info(f"{'Task':<20} {'Model':<16} {'LOSO Acc':>10} {'Std':>8} {'F1':>8}")
    log.info("-" * 70)
    for s in all_summaries:
        log.info(f"{s['task']:<20} {s['model']:<16} "
                 f"{s['loso_acc_mean']:>10.4f} {s['loso_acc_std']:>8.4f} {s['loso_f1_mean']:>8.4f}")
    log.info("=" * 70)

    write_combined_csv()


def write_combined_csv(results_root="results", out_path=None):
    """
    Combine every ``results/<task>/<model>_loso.json`` into one CSV with **one row per
    individual LOSO fold** (i.e. per held-out session), written to ``results/summary.csv``.

    Each row carries the fold's own metrics (accuracy, balanced accuracy, weighted F1,
    Cohen's kappa, loss) plus the ``(task, model)`` LOSO aggregate as context columns, so
    the file holds both the per-fold detail and the summary in one place. Built from
    on-disk JSON, so it captures all completed runs even across separate invocations.
    """
    out_path = out_path or os.path.join(results_root, "summary.csv")
    fields = ["task", "model", "subject", "session",
              "accuracy", "balanced_accuracy", "f1_score", "kappa", "loss",
              "n_subjects", "loso_acc_mean", "loso_acc_std",
              "loso_balacc_mean", "loso_f1_mean", "loso_kappa_mean"]

    def _r(x):
        return round(x, 6) if isinstance(x, (int, float)) else float("nan")

    rows = []
    for jf in sorted(glob.glob(os.path.join(results_root, "*", "*_loso.json"))):
        with open(jf) as f:
            d = json.load(f)
        for stem, m in d.get("per_session", {}).items():
            subject, _, session = stem.partition("_")
            rows.append({
                "task": d["task"],
                "model": d["model"],
                "subject": subject.replace("sub-", ""),
                "session": session.replace("ses-", ""),
                "accuracy": _r(m.get("accuracy")),
                "balanced_accuracy": _r(m.get("balanced_accuracy")),
                "f1_score": _r(m.get("f1_score")),
                "kappa": _r(m.get("kappa")),
                "loss": _r(m.get("loss")),
                "n_subjects": d["n_subjects"],
                "loso_acc_mean": _r(d.get("loso_acc_mean")),
                "loso_acc_std": _r(d.get("loso_acc_std")),
                "loso_balacc_mean": _r(d.get("loso_balacc_mean")),
                "loso_f1_mean": _r(d.get("loso_f1_mean")),
                "loso_kappa_mean": _r(d.get("loso_kappa_mean")),
            })

    if not rows:
        log.warning("[CSV] No result JSON files found — nothing to combine.")
        return

    rows.sort(key=lambda r: (r["task"], r["model"], r["subject"], r["session"]))
    os.makedirs(results_root, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"[CSV] Wrote combined per-fold summary ({len(rows)} rows) → {out_path}")


if __name__ == "__main__":
    main()
