"""
Wilcoxon signed-rank significance tests for the EEG speech-decoding benchmark.

Two families of test, both computed from the per-subject LOSO scores stored in
``results/<task>/<model>_loso.json`` (no retraining, no re-inference):

  1. **Against chance** — for each (task, model), a one-sample Wilcoxon signed-rank
     test asking whether the 14 per-subject accuracies lie *above* the chance level
     (1 / n_classes). One-sided (alternative="greater").

  2. **Pairwise between models** — for each task, a paired (two-sided) Wilcoxon
     signed-rank test on the per-subject accuracies of two models, aligned by subject.
     We report every model against EEGNet (the strongest baseline), which is the
     comparison the paper makes.

Cohen's kappa is also tested against 0 (its chance value) as a cross-check, since
kappa = 0 corresponds exactly to chance agreement.

Outputs:
  * a human-readable table to stdout,
  * ``results/wilcoxon_tests.csv`` with every test,
  * a ready-to-paste LaTeX snippet to stdout.

Run:
    python -m src.analysis.wilcoxon_tests
    # or: python src/analysis/wilcoxon_tests.py
"""
import os
import csv
import json

import numpy as np
from scipy.stats import wilcoxon

RESULTS_DIR = "results"
OUT_CSV = os.path.join(RESULTS_DIR, "wilcoxon_tests.csv")

# (task_key, display title, n_classes) — chance = 1 / n_classes.
TASKS = [
    ("speech_mode",       "Speech mode",       3),
    ("semantic_category", "Semantic category", 6),
    ("word",              "Word identity",     60),
]

# (json model name, display label).
MODELS = [
    ("EEGNetv4",        "EEGNet"),
    ("ShallowFBCSPNet", "ShallowNet"),
    ("Deep4Net",        "Deep4Net"),
    ("EEGConformer",    "Conformer"),
    ("LaBraM",          "LaBraM"),
    ("EEGMamba",        "EEGMamba"),
]

REFERENCE_MODEL = "EEGNetv4"  # baseline that every other model is compared against
ALPHA = 0.05


def _load_subject_map(task, model_json, key):
    """Return {subject: score} for one (task, model), or {} if the JSON is absent."""
    path = os.path.join(RESULTS_DIR, task, f"{model_json}_loso.json")
    if not os.path.exists(path):
        return {}
    d = json.load(open(path))
    return {s: float(v) for s, v in d.get(key, {}).items()}


def _stars(p):
    if p is None or np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _wilcoxon_greater(values, baseline):
    """One-sided Wilcoxon signed-rank: are `values` greater than a constant `baseline`?

    Returns (statistic, p_value, n_nonzero). If every difference is zero (e.g. a model
    pinned exactly at chance) the test is undefined and we return p = 1.0.
    """
    diffs = np.asarray(values) - baseline
    nz = int(np.count_nonzero(diffs))
    if nz == 0:
        return float("nan"), 1.0, 0
    # zero_method="wilcox" drops zero-differences, matching the classic test.
    stat, p = wilcoxon(diffs, alternative="greater", zero_method="wilcox")
    return float(stat), float(p), nz


def _wilcoxon_paired(a, b):
    """Two-sided paired Wilcoxon signed-rank between aligned vectors a and b."""
    diffs = np.asarray(a) - np.asarray(b)
    nz = int(np.count_nonzero(diffs))
    if nz == 0:
        return float("nan"), 1.0, 0
    stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return float(stat), float(p), nz


def run():
    rows = []

    # ── 1. Against chance (accuracy and kappa) ────────────────────────────────
    print("\n=== Wilcoxon signed-rank vs. chance (one-sided, alternative='greater') ===")
    print(f"{'Task':<18}{'Model':<12}{'n':>3}  {'mean acc':>9}{'chance':>8}"
          f"{'p(acc)':>10}{'':>5}{'p(kappa>0)':>12}{'':>5}")
    for task, title, n_cls in TASKS:
        chance = 1.0 / n_cls
        for mj, label in MODELS:
            acc = _load_subject_map(task, mj, "per_subject_acc")
            kap = _load_subject_map(task, mj, "per_subject_kappa")
            if not acc:
                continue
            acc_vals = list(acc.values())
            _, p_acc, n_nz = _wilcoxon_greater(acc_vals, chance)
            p_kappa = float("nan")
            if kap:
                _, p_kappa, _ = _wilcoxon_greater(list(kap.values()), 0.0)
            mean_acc = float(np.mean(acc_vals))
            print(f"{title:<18}{label:<12}{len(acc_vals):>3}  {mean_acc*100:>8.1f}%"
                  f"{chance*100:>7.1f}%{p_acc:>10.4f}{_stars(p_acc):>5}"
                  f"{p_kappa:>12.4f}{_stars(p_kappa):>5}")
            rows.append({
                "test": "vs_chance",
                "task": task,
                "model": label,
                "model_b": "",
                "n": len(acc_vals),
                "metric": "accuracy",
                "mean_a": round(mean_acc, 6),
                "mean_b": round(chance, 6),
                "p_value": round(p_acc, 6),
                "sig": _stars(p_acc),
                "p_kappa_vs0": round(p_kappa, 6) if not np.isnan(p_kappa) else "",
            })

    # ── 2. Pairwise vs the reference baseline (EEGNet), accuracy ──────────────
    ref_label = dict(MODELS)[REFERENCE_MODEL]
    print(f"\n=== Pairwise Wilcoxon vs. {ref_label} (two-sided, paired by subject, accuracy) ===")
    print(f"{'Task':<18}{'Model':<12}{'n':>3}  {'mean':>8}{'ref mean':>10}"
          f"{'delta':>8}{'p':>10}{'':>5}")
    for task, title, n_cls in TASKS:
        ref = _load_subject_map(task, REFERENCE_MODEL, "per_subject_acc")
        for mj, label in MODELS:
            if mj == REFERENCE_MODEL:
                continue
            other = _load_subject_map(task, mj, "per_subject_acc")
            if not other or not ref:
                continue
            subjects = sorted(set(ref) & set(other))
            if len(subjects) < 3:
                continue
            a = [other[s] for s in subjects]
            b = [ref[s] for s in subjects]
            _, p, _ = _wilcoxon_paired(a, b)
            ma, mb = float(np.mean(a)), float(np.mean(b))
            print(f"{title:<18}{label:<12}{len(subjects):>3}  {ma*100:>7.1f}%"
                  f"{mb*100:>9.1f}%{(ma-mb)*100:>+7.1f}%{p:>10.4f}{_stars(p):>5}")
            rows.append({
                "test": "pairwise_vs_ref",
                "task": task,
                "model": label,
                "model_b": ref_label,
                "n": len(subjects),
                "metric": "accuracy",
                "mean_a": round(ma, 6),
                "mean_b": round(mb, 6),
                "p_value": round(p, 6),
                "sig": _stars(p),
                "p_kappa_vs0": "",
            })

    # ── Write CSV ─────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fields = ["test", "task", "model", "model_b", "n", "metric",
              "mean_a", "mean_b", "p_value", "sig", "p_kappa_vs0"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[CSV] wrote {len(rows)} tests → {OUT_CSV}")
    return rows


if __name__ == "__main__":
    run()
