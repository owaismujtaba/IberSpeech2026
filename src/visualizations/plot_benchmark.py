"""
Publication-quality benchmark figure for the EEG speech-decoding study.

For a Nature-type article the key message is *distributional*: how each model's
per-subject leave-one-subject-out (LOSO) accuracy is spread, and how close it sits
to chance, across three tasks of increasing linguistic difficulty. A bare bar chart
hides the large inter-subject variability that dominates EEG decoding, so we instead
show, per task and per model, a box of the per-subject scores overlaid with the
individual subjects as jittered dots, plus an explicit chance line.

Design follows common Nature figure conventions: compact double-column width,
sans-serif ~7 pt type, de-cluttered axes (no top/right spines), a colour-blind-safe
palette (Wong 2011), the mean marked, and vector PDF + 300-dpi PNG output.

Run:
    python -m src.visualizations.plot_benchmark
    # or: python src/visualizations/plot_benchmark.py
"""
import os
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
OUT_DIR = os.path.join("results", "figures")

# ── What to plot ─────────────────────────────────────────────────────────────
# (task_key, panel title, n_classes) — chance = 1 / n_classes.
TASKS = [
    ("speech_mode",       "Speech mode\n(3 classes)",       3),
    ("semantic_category", "Semantic category\n(6 classes)", 6),
    ("word",              "Word identity\n(60 classes)",    60),
]

# (json model name, display label, is_foundation).
MODELS = [
    ("EEGNetv4",        "EEGNet",    False),
    ("ShallowFBCSPNet", "ShallowNet", False),
    ("Deep4Net",        "Deep4Net",  False),
    ("EEGConformer",    "Conformer", False),
    ("LaBraM",          "LaBraM",    True),
    ("EEGMamba",        "EEGMamba",  True),
]

# Colour-blind-safe palette (Wong, Nature Methods 2011).
C_BASELINE   = "#0072B2"  # blue
C_FOUNDATION = "#D55E00"  # vermillion
C_CHANCE     = "#999999"
C_POINT      = "#333333"

METRIC = "accuracy"  # uses per_subject_acc; set to "kappa" for per_subject_kappa


def _nature_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,  # editable text in vector output
        "ps.fonttype": 42,
    })


def _subject_key(metric):
    return {"accuracy": "per_subject_acc",
            "kappa": "per_subject_kappa",
            "balanced_accuracy": "per_subject_balacc",
            "f1_score": "per_subject_f1"}[metric]


def load_per_subject(task, model_json, metric=METRIC):
    """Return a list of per-subject scores for one (task, model), or [] if absent."""
    path = os.path.join(RESULTS_DIR, task, f"{model_json}_loso.json")
    if not os.path.exists(path):
        return []
    d = json.load(open(path))
    vals = d.get(_subject_key(metric), {})
    return [float(v) for v in vals.values()]


def make_figure(metric=METRIC, out_dir=OUT_DIR, layout="column"):
    """Render the benchmark figure.

    layout="column"  → single-column figure: tasks stacked vertically (3 rows × 1),
                       sized for one column of a two-column template (~88 mm wide).
    layout="row"     → double-column figure: tasks side by side (1 row × 3).
    """
    _nature_style()
    os.makedirs(out_dir, exist_ok=True)

    as_pct = metric in ("accuracy", "balanced_accuracy", "f1_score")
    scale = 100.0 if as_pct else 1.0
    ylabel = {"accuracy": "LOSO acc. (%)",
              "kappa": "Cohen's κ",
              "balanced_accuracy": "Bal. acc. (%)",
              "f1_score": "Weighted F1 (×100)"}[metric]

    n_panels = len(TASKS)
    if layout == "column":
        # Single column ~88 mm = 3.46 in; stack the three tasks vertically.
        fig, axes = plt.subplots(n_panels, 1, figsize=(3.4, 5.4),
                                 constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, n_panels, figsize=(7.2, 2.5),
                                 constrained_layout=True)

    rng = np.random.default_rng(0)
    labels = [lbl for _, lbl, _ in MODELS]
    colors = [C_FOUNDATION if found else C_BASELINE for _, _, found in MODELS]
    xs = np.arange(len(MODELS))
    bottom_ax = axes[-1] if layout == "column" else None

    for i, (ax, (task, title, n_cls)) in enumerate(zip(axes, TASKS)):
        chance = (1.0 / n_cls) * scale

        data = [np.array(load_per_subject(task, mj, metric)) * scale
                for mj, _, _ in MODELS]

        # Box (distribution) per model — thin, no fliers (points drawn separately).
        bp = ax.boxplot(
            data, positions=xs, widths=0.55, showfliers=False,
            patch_artist=True, medianprops=dict(color="black", linewidth=1.0),
            whiskerprops=dict(linewidth=0.6), capprops=dict(linewidth=0.6),
            boxprops=dict(linewidth=0.6),
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.30)
            patch.set_edgecolor(c)

        # Individual subjects as jittered dots.
        for x, vals, c in zip(xs, data, colors):
            if len(vals) == 0:
                continue
            jit = rng.uniform(-0.13, 0.13, size=len(vals))
            ax.scatter(x + jit, vals, s=6, color=C_POINT, alpha=0.7,
                       linewidths=0, zorder=3)
            # Mean marker (diamond).
            ax.scatter([x], [vals.mean()], marker="D", s=16, color=c,
                       edgecolor="black", linewidths=0.5, zorder=4)

        # Chance line.
        ax.axhline(chance, ls=(0, (4, 3)), color=C_CHANCE, linewidth=0.8, zorder=1)
        ax.text(len(MODELS) - 0.5, chance, " chance", va="center", ha="left",
                color=C_CHANCE, fontsize=6, clip_on=False)

        # Compact one-line panel title (works when stacked).
        ax.set_title(title.replace("\n", " "), pad=4, fontsize=7.5)
        ax.set_xticks(xs)
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(y=0.10)

        # In the stacked layout, only the bottom panel carries x tick labels.
        if layout == "column" and ax is not bottom_ax:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(labels, rotation=35, ha="right")

    if layout == "row":
        for ax in axes[1:]:
            ax.set_ylabel("")
        axes[0].set_ylabel(ylabel)

    # Shared legend for the baseline/foundation colour coding.
    legend_handles = [
        Patch(facecolor=C_BASELINE, alpha=0.30, edgecolor=C_BASELINE, label="CNN baseline"),
        Patch(facecolor=C_FOUNDATION, alpha=0.30, edgecolor=C_FOUNDATION, label="Foundation model"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="grey",
                   markeredgecolor="black", markersize=5, label="Mean"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_POINT,
                   markersize=4, label="Subject"),
    ]
    if layout == "column":
        # Reserve headroom above the top panel so the legend never overlaps its title.
        fig.legend(handles=legend_handles, loc="lower center", ncol=2,
                   frameon=False, bbox_to_anchor=(0.5, 1.005))
    else:
        fig.legend(handles=legend_handles, loc="upper center", ncol=4,
                   frameon=False, bbox_to_anchor=(0.5, 1.04))

    suffix = "" if layout == "column" else "_wide"
    stem = os.path.join(out_dir, f"benchmark_per_subject_{metric}{suffix}")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[FIG] wrote {stem}.pdf and {stem}.png  (layout={layout})")
    return f"{stem}.pdf"


if __name__ == "__main__":
    # Single-column figures for the two-column template (default).
    make_figure("accuracy", layout="column")
    make_figure("kappa", layout="column")
