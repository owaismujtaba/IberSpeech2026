# Does EEG Foundation Models Transfer to Speech?

**A benchmark of EEG foundation models against convolutional baselines for overt, covert and imagined speech decoding.**

This repository contains the code and results for the paper
*"Does EEG Foundation Models Transfer to Speech? A Benchmark on Overt and Imagined
Speech Decoding"* (IberSpeech 2026). We ask whether the large gains EEG foundation
models report on motor imagery, seizure and sleep tasks carry over to the much harder
problem of **speech and language decoding**.

We compare two pre-trained foundation models against four convolutional baselines under a
single unified preprocessing + fine-tuning pipeline, with a strict
**Leave-One-Subject-Out (LOSO)** protocol.

| Family | Model | Params | Source |
|--------|-------|:------:|--------|
| Baseline | **EEGNetv4** | ~16.6K | `braindecode` (from scratch) |
| Baseline | **ShallowFBCSPNet** | ~162K | `braindecode` (from scratch) |
| Baseline | **Deep4Net** | ~0.3M | `braindecode` (from scratch) |
| Baseline | **EEGConformer** | ~406K | `braindecode` (from scratch) |
| Foundation | **LaBraM-Base** | ~5.8M | HuggingFace, fine-tuned |
| Foundation | **EEGMamba** | ~8.3M | HuggingFace, fine-tuned |

**Headline finding:** neither foundation model outperforms a 16K-parameter EEGNet.
Despite ~350–500× more parameters and thousands of hours of pre-training, large-scale
general-purpose EEG pre-training confers **no consistent advantage** for speech decoding.

## Corpora

This repository operates on **UGR-MINDVOICE** only. The second corpus reported in the
paper, **BCI Competition 2020 Track 3** (imagined speech, 5 classes), was processed
separately by a co-author and is **not part of this repository** — its numbers are
reported in the paper directly.

**UGR-MINDVOICE** is a BIDS-compliant corpus of 64-channel EEG (10–20, 1000 Hz, FCz
reference) of Iberian-Spanish overt and covert speech. The raw data live under `BIDS/`.
Three decoding tasks of increasing linguistic difficulty are defined on it:

| Task                | Classes | Chance | Description                          |
|---------------------|:-------:|:------:|--------------------------------------|
| `speech_mode`       |   3     | 33.3%  | Overt vs Covert vs Rest              |
| `semantic_category` |   6     | 16.7%  | Semantic category of the spoken word |
| `word`              |   60    |  1.7%  | The 60 real Spanish words            |

Semantic categories come from `Sem Categories.xlsx` (6 categories × 10 words).

### Label construction

Only **Experiment** trials of the **Words** task are used (Practice trials excluded):

- **Speech mode** is read from the annotation prefix (`Real → Covert`, `Silent → Overt`).
  **Rest** epochs are taken from non-speech `StartFixation` onsets and subsampled to match
  the Overt count, balancing the three classes (~255 each per session).
- **`semantic_category`** and **`word`** use the 60 real words only; pseudowords and
  `silence` are dropped. Both Overt and Covert utterances of each word are **pooled into a
  single label**, since these tasks target linguistic content, not the speaking mode (the
  overt/covert contrast is captured separately by `speech_mode`).

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` covers the core stack (`mne`, `mne-bids`, `torch`, `braindecode`,
`scikit-learn`, `pandas`, `pyyaml`, `tqdm`, `matplotlib`). A few extras are also needed:

```bash
pip install einops huggingface_hub scipy openpyxl
```

- `einops` / `huggingface_hub` — used by the EEGMamba implementation and to pull the
  pretrained LaBraM / EEGMamba checkpoints.
- `scipy` — Wilcoxon significance tests.
- `openpyxl` — to read `Sem Categories.xlsx`.

The pretrained foundation-model weights are downloaded automatically on first use:

- **LaBraM-Base** — `braindecode/Labram-Braindecode` (HuggingFace).
- **EEGMamba** — `weighting666/EEGMamba` (HuggingFace).

The classification head (and, for LaBraM, a few positional embeddings) is left at fresh
init because our class counts and epoch length differ from pre-training; the rest of the
backbone is loaded from the checkpoint.

> Examples below call Python as `python`. Use whatever interpreter has the dependencies
> installed (this project was developed against a conda env named `libri`).

## Configuration

Everything is driven by `config.yaml`:

- `dataset.bids_root` / `dataset.subjects` — BIDS root and the subjects (LOSO is run over
  these 14 subjects).
- `preprocessing` — band-pass (0.1–75 Hz), 50 Hz notch, resample to 200 Hz, epoch window
  (−0.2 to 1.5 s), baseline.
- `decoding.tasks` — any subset of `[speech_mode, semantic_category, word]`.
- `models` — list of architectures to train and compare
  (`EEGNetv4`, `EEGConformer`, `ShallowFBCSPNet`, `Deep4Net`, `LaBraM`, `EEGMamba`).
- `training` — `batch_size`, `epochs` (100, early-stopping patience 10), `lr`,
  `weight_decay`, `device`, global `seed`.
- `workflow.create_words_epochs` — set `true` **once** to (re)build epochs from raw BIDS.

### Model-specific input scaling

Normalization is chosen per model in `src/engine/trainer.py`:

- **Foundation models (LaBraM, EEGMamba)** → **physical** scaling (µV / 100), matching the
  unit their public checkpoints were pre-trained on. Z-scoring shifts the input
  distribution away from what the pretrained weights expect.
- **Baselines** → per-epoch, per-channel **z-score**.

Loss is `NLLLoss` for the `braindecode` models that output log-softmax (Deep4Net,
ShallowFBCSPNet, EEGConformer) and `CrossEntropyLoss` otherwise.

## Usage

### 1. Build epochs (once)

Set `workflow.create_words_epochs: true` in `config.yaml`, then:

```bash
python main.py
```

This preprocesses each subject's raw EEG (filter → notch → ICA → average reference →
resample), extracts Words speech (Overt/Covert) and balanced Rest epochs, and saves
`processed/words/sub-XX_ses-YY.fif` with a metadata table (`mode`, `word`, `category`).
ICA runs per subject, so this step is slow (a few minutes per session).

### 2. Train + evaluate (LOSO)

Set `workflow.create_words_epochs: false` and run again:

```bash
python main.py
```

For every `(task × model)` combination it runs all LOSO folds (one held-out subject per
fold) and:

- writes per-fold and aggregate metrics to `results/<task>/<model>_loso.json`
  (flushed after **every** fold, so partial runs are never lost);
- saves each fold's best checkpoint to `saved_models/<task>/<model>/sub-XX.pt`
  (state dict + `n_chans` / `n_classes` / `n_times` + label `vocab`, so any fold's model
  can be reloaded standalone);
- writes per-subject / subject-vs-session bar plots (`*.png`);
- rebuilds `results/summary.csv` (one row per LOSO fold across all tasks/models).

Reported metrics: **accuracy, balanced accuracy, weighted F1, Cohen's κ**, as mean ± SD of
the per-subject scores across the 14 LOSO folds.

## Analysis scripts (no retraining)

These reuse the saved checkpoints / result JSONs — no models are retrained.

```bash
# Wilcoxon signed-rank tests: each model vs chance (one-sided) and
# pairwise vs EEGNet (paired two-sided). → results/wilcoxon_tests.csv
python -m src.analysis.wilcoxon_tests

# Overt → covert degradation: re-evaluate each pooled-trained LOSO checkpoint
# separately on the held-out subject's overt-only vs covert-only epochs.
# → results/overt_vs_covert/<task>_<model>.json + summary.csv
CUDA_VISIBLE_DEVICES="" python -m src.analysis.overt_vs_covert

# Backfill kappa / balanced accuracy into older result JSONs that predate
# those metrics, by re-running inference on the saved checkpoints.
python recompute_kappa.py
```

## Project structure

```
config.yaml                      # single source of truth for a run
main.py                          # orchestrates task × model × LOSO; writes JSON/CSV/plots
recompute_kappa.py               # backfill κ / balanced-acc into old result JSONs
paper.tex                        # the IberSpeech 2026 manuscript
BIDS/                            # raw UGR-MINDVOICE data (BIDS)
processed/words/                 # cached *.fif epochs + metadata (built once)
saved_models/<task>/<model>/     # per-fold checkpoints (sub-XX.pt)
results/                         # *_loso.json, summary.csv, wilcoxon_tests.csv, plots
figures/                         # paper figures

src/
  data/
    bids_loader.py               # BIDS loading + Words speech/rest event extraction
    labels.py                    # word → semantic-category map from the spreadsheet
    create_epochs.py             # preprocessing + epoching with metadata
    dataset.py                   # task label selection + LOSO fold construction
  preprocess/preprocess.py       # filter / ICA / reference / epoch creation
  engine/
    models.py                    # model factory (braindecode + LaBraM/EEGMamba)
    trainer.py                   # training/eval loop, early stopping, input scaling
    metrics.py                   # accuracy, balanced accuracy, weighted F1, Cohen's κ
    plot_results.py              # per-subject / subject-vs-session bar plots
  analysis/
    wilcoxon_tests.py            # significance tests from per-subject scores
    overt_vs_covert.py           # overt vs covert degradation (inference only)
  visualizations/plot_benchmark.py
  utils/                         # config parser, logger, seeding, IO helpers
```

## License

See [`LICENSE`](LICENSE).
