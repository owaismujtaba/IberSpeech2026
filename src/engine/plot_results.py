import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np


def _parse_stem(stem):
    """Extract subject and session from filename stem like 'sub-01_ses-1'."""
    m = re.match(r"sub-(\w+)_ses-(\w+)", stem)
    if m:
        return f"sub-{m.group(1)}", f"ses-{m.group(2)}"
    return stem, ""


def plot_per_subject(results: dict, metric: str = "accuracy", title: str = "",
                     save_path: str = None):
    """Bar plot of per-subject-session performance."""
    stems = sorted(results.keys())
    values = [results[s][metric] for s in stems]
    labels = [f"{_parse_stem(s)[0]}\n{_parse_stem(s)[1]}" for s in stems]

    fig, ax = plt.subplots(figsize=(max(8, len(stems) * 0.7), 5))
    colors = plt.cm.tab20(np.linspace(0, 1, len(stems)))
    bars = ax.bar(range(len(stems)), values, color=colors, edgecolor="black", linewidth=0.5)

    mean_val = np.mean(values)
    ax.axhline(mean_val, color="red", linestyle="--", linewidth=1.2, label=f"Mean: {mean_val:.3f}")

    ax.set_xticks(range(len(stems)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_ylim(0, 1.05)
    ax.set_title(title or f"Per-Subject-Session {metric.replace('_', ' ').title()}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    plt.show()
    return fig


def plot_subject_vs_session(results: dict, metric: str = "accuracy",
                             title: str = "", save_path: str = None):
    """Grouped bar plot — subjects on x-axis, sessions as groups."""
    subject_sessions = {}
    for stem in results:
        sub, ses = _parse_stem(stem)
        subject_sessions.setdefault(sub, {})[ses] = results[stem][metric]

    subjects = sorted(subject_sessions.keys())
    all_sessions = sorted({ses for v in subject_sessions.values() for ses in v})
    x = np.arange(len(subjects))
    width = 0.8 / max(len(all_sessions), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(subjects) * 0.9), 5))
    for i, ses in enumerate(all_sessions):
        vals = [subject_sessions[sub].get(ses, np.nan) for sub in subjects]
        offset = (i - len(all_sessions) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.9, label=ses)

    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_ylim(0, 1.05)
    ax.set_title(title or f"Per-Subject {metric.replace('_', ' ').title()} by Session")
    ax.legend(title="Session")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    plt.show()
    return fig


def load_and_plot(json_path: str, metric: str = "accuracy"):
    """Load a saved per-subject JSON and generate both plots."""
    with open(json_path) as f:
        results = json.load(f)

    base = os.path.splitext(json_path)[0]
    model_tag = os.path.basename(base)

    plot_per_subject(
        results, metric=metric,
        title=f"{model_tag} — Per-Subject-Session {metric}",
        save_path=f"{base}_bar.png",
    )
    plot_subject_vs_session(
        results, metric=metric,
        title=f"{model_tag} — {metric} by Subject and Session",
        save_path=f"{base}_grouped.png",
    )
    return results
