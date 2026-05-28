# Evaluator for Braindecode Models on UGR-MINDVOICE (BIDS)

This repository provides a modular PyTorch framework to load EEG data from a BIDS-compliant dataset, preprocess it, and evaluate foundation/deep learning models from the `braindecode` library.

## Requirements

The dependencies can be installed via the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Configuration

All dataset paths, preprocessing parameters, and training hyperparameters are defined in `config.yaml`.
- Ensure `dataset.bids_root` points to your correct BIDS directory.
- `model.name` specifies the architecture to use (e.g., `EEGConformer`, `EEGNetv4`, `ShallowFBCSPNet`).

## Usage

You can run the full pipeline using the main entry script:

```bash
cd eeg_eval
python main.py --config config.yaml
```

You can override configuration parameters via command-line arguments:
```bash
python main.py --model EEGNetv4 --batch_size 64 --epochs 100
```

## Project Structure

- `data/bids_loader.py`: Handles dynamic loading of BIDS subjects and sessions using `mne-bids`.
- `data/preprocess.py`: Applies filtering, resampling, and epoching to continuous MNE raw files.
- `data/dataset.py`: A `torch.utils.data.Dataset` wrapper for MNE Epochs.
- `models/model_factory.py`: Dynamically instantiates models from `braindecode.models`.
- `engine/trainer.py`: Standard PyTorch training and evaluation loops.
- `engine/metrics.py`: Computes Accuracy and F1 scores.
