import yaml
import argparse

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_args():
    parser = argparse.ArgumentParser(description="EEG Foundational Models Evaluation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default=None, help="Model to evaluate (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides config)")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda) (overrides config)")
    return parser.parse_args()

def merge_args_with_config(args, config):
    if args.model:
        config['model']['name'] = args.model
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.device:
        config['training']['device'] = args.device
    return config
