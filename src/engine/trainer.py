import os

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from .metrics import compute_metrics
from src.engine.models import create_model
from src.utils.logger import get_logger
from src.utils.seed import set_seed

log = get_logger()

# LaBraM and EEGMamba were pre-trained on signals in the physical "0.1 mV" unit
# (microvolts / 100), not on per-channel z-scored inputs. Their fine-tuning recipes
# feed data/100 on µV signals (see the original EEGMamba and LaBraM repos), so z-scoring
# shifts the input distribution away from what the pretrained weights expect. We therefore
# scale these models to their native unit and z-score only the from-scratch baselines.
PHYSICAL_SCALE_MODELS = {"LaBraM", "EEGMamba"}


def zscore_normalize(x: np.ndarray) -> np.ndarray:
    """Z-score each epoch per channel: (x - mean) / std  over the time axis."""
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1, keepdims=True) + 1e-8
    return (x - mean) / std


def physical_scale(x: np.ndarray) -> np.ndarray:
    """Volts → the 0.1 mV unit (µV / 100) used to pre-train LaBraM and EEGMamba."""
    return (x * 1e6 / 100.0).astype(np.float32)


def normalize_epochs(x: np.ndarray, scheme: str) -> np.ndarray:
    return physical_scale(x) if scheme == "physical" else zscore_normalize(x)


def make_loader(x, y, batch_size=64, shuffle=True, tag="", norm="zscore"):
    before_min, before_max = x.min(), x.max()
    x = normalize_epochs(x, norm)
    log.debug(
        f"[LOADER] {tag}  norm={norm}  raw range=[{before_min:.6e}, {before_max:.6e}]  "
        f"out range=[{x.min():.3f}, {x.max():.3f}]  std={x.std():.4f}  "
        f"shape={x.shape}  n_labels={len(np.unique(y))}  batch_size={batch_size}"
    )
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train(config, X_train, y_train, val_dict, n_classes, tag="", save_path=None, vocab=None,
          seed=42, ch_names=None):
    """
    Train one model on pre-extracted arrays and evaluate per held-out session.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training epochs ``(N, C, T)`` and integer labels (already mapped to the
        task's fixed 0-based vocabulary).
    val_dict : dict
        ``{session_stem: (X, y)}`` for the held-out validation subject.
    n_classes : int
        Size of the task vocabulary (model output dimension).
    tag : str
        Short identifier for logging (e.g. ``"speech_mode/EEGNetv4/sub-03"``).
    """
    model_name = config['model']['name']

    set_seed(seed)  # reproducible weight init, data shuffling and dropout for this fold

    log.info("=" * 60)
    log.info(f"[TRAIN] Starting  model={model_name}  {tag}  seed={seed}")

    train_data   = X_train
    train_labels = y_train
    all_val_labels = np.concatenate([y for _, y in val_dict.values()])

    n_chans = train_data.shape[1]
    n_times = train_data.shape[2]
    log.info(
        f"[TRAIN] n_classes={n_classes}  n_chans={n_chans}  n_times={n_times}  "
        f"train={train_data.shape[0]} epochs  "
        f"unique_train_labels={len(np.unique(train_labels))}  "
        f"unique_val_labels={len(np.unique(all_val_labels))}"
    )

    # ── Hyperparameters ───────────────────────────────────────────────────────
    batch_size   = config['training']['batch_size']
    max_epochs   = config['training']['epochs']
    lr           = config['training']['lr']
    weight_decay = config['training']['weight_decay']
    patience     = config['training'].get('patience', 10)
    log.info(
        f"[TRAIN] Hyperparams  →  batch_size={batch_size}  max_epochs={max_epochs}  "
        f"lr={lr}  weight_decay={weight_decay}  patience={patience}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"[TRAIN] Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = create_model(config, n_chans, n_classes, n_times, ch_names=ch_names)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"[TRAIN] Model '{model_name}'  →  trainable parameters={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Braindecode models with add_log_softmax=True output log-probabilities — use NLLLoss.
    # Custom models (LaBraM, EEGMamba) output raw logits — use CrossEntropyLoss.
    _LOG_SOFTMAX_MODELS = {'Deep4Net', 'ShallowFBCSPNet', 'EEGConformer'}
    _uses_log_softmax = model_name in _LOG_SOFTMAX_MODELS
    criterion = nn.NLLLoss() if _uses_log_softmax else nn.CrossEntropyLoss()
    log.info(f"[TRAIN] Loss function: {'NLLLoss' if _uses_log_softmax else 'CrossEntropyLoss'}")

    # ── Input normalization (model-specific) ──────────────────────────────────
    norm_scheme = "physical" if model_name in PHYSICAL_SCALE_MODELS else "zscore"
    log.info(f"[TRAIN] Input normalization: {norm_scheme} "
             f"({'µV/100 to match pretraining' if norm_scheme == 'physical' else 'per-epoch per-channel z-score'})")

    # ── Data loaders ──────────────────────────────────────────────────────────
    train_loader = make_loader(train_data, train_labels, batch_size=batch_size,
                               shuffle=True, tag="TRAIN", norm=norm_scheme)
    log.info(f"[TRAIN] Train loader  →  {len(train_loader)} batches  "
             f"({len(train_loader.dataset)} samples)")

    val_loaders = {}
    for stem, (vd, vl) in val_dict.items():
        val_loaders[stem] = make_loader(vd, vl.astype(np.int64), batch_size=batch_size,
                                        shuffle=False, tag=f"VAL/{stem}", norm=norm_scheme)

    all_val_data = np.concatenate([vd for vd, _ in val_dict.values()])
    val_loader = make_loader(all_val_data, all_val_labels.astype(np.int64),
                             batch_size=batch_size, shuffle=False, tag="VAL/aggregate", norm=norm_scheme)
    log.info(f"[TRAIN] Aggregate val loader  →  {len(val_loader)} batches  "
             f"({len(val_loader.dataset)} samples)")

    if train_data.shape[1:] != all_val_data.shape[1:]:
        log.error("[TRAIN]  *** Shape mismatch between train and val (channels or time points differ)! ***")

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {"train": [], "val": []}
    best_val_acc = -float("inf")
    best_state = None
    epochs_no_improve = 0

    log.info(f"[TRAIN] Starting epoch loop  (max_epochs={max_epochs}  patience={patience})")

    for epoch in range(max_epochs):
        log.info(f"[TRAIN] Epoch {epoch + 1}/{max_epochs}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics   = evaluate(model, val_loader, criterion, device)

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        val_acc = val_metrics.get("accuracy", 0.0)
        log.info(
            f"[TRAIN]   train  loss={train_metrics['loss']:.4f}  "
            f"acc={train_metrics.get('accuracy', 0.0):.4f}  "
            f"f1={train_metrics.get('f1_score', 0.0):.4f}"
        )
        log.info(
            f"[TRAIN]   val    loss={val_metrics['loss']:.4f}  "
            f"acc={val_acc:.4f}  "
            f"f1={val_metrics.get('f1_score', 0.0):.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            log.debug(f"[TRAIN]   New best val_acc={best_val_acc:.4f} — checkpoint saved")
        else:
            epochs_no_improve += 1
            log.debug(f"[TRAIN]   No improvement  ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                log.info(f"[TRAIN] Early stopping at epoch {epoch + 1}  (patience={patience}  best_val_acc={best_val_acc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    log.info(f"[TRAIN] Training done  →  best_val_acc={best_val_acc:.4f}")

    # ── Save best checkpoint ──────────────────────────────────────────────────
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "n_chans": n_chans,
            "n_classes": n_classes,
            "n_times": n_times,
            "vocab": vocab,
            "best_val_acc": best_val_acc,
            "tag": tag,
        }, save_path)
        log.info(f"[TRAIN] Saved checkpoint → {save_path}")

    # ── Per held-out session evaluation ───────────────────────────────────────
    per_subject_results = {}
    for stem, loader in val_loaders.items():
        metrics = evaluate(model, loader, criterion, device)
        per_subject_results[stem] = metrics
        log.info(f"[TRAIN]   {stem}  acc={metrics['accuracy']:.4f}  f1={metrics['f1_score']:.4f}")

    log.info("=" * 60)
    return model, history, per_subject_results


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        if outputs.ndim > 2:
            outputs = outputs.flatten(1)  # (B, seq, C) → (B, seq*C); safe for any batch size

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": total_loss / (batch_idx + 1)})

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    return metrics


def evaluate(model, dataloader, criterion, device, return_preds=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            if outputs.ndim > 2:
                outputs = outputs.flatten(1)  # (B, seq, C) → (B, seq*C); safe for any batch size

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    if return_preds:
        return metrics, all_labels, all_preds
    return metrics