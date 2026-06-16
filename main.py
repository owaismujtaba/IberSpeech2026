import os

from src.utils.config_parser import load_config

from src.data.create_epochs import create_epochs
from src.data.dataset import load_and_split_per_file

# pyrefly: ignore [missing-import]
from src.engine.trainer import train

import pdb


config = load_config()

create_syllable = config['workflow'].get('create_syllable_epochs', False)
create_words = config['workflow'].get('create_words_epochs', False)

if create_syllable or create_words:
    create_epochs(create_syllable=create_syllable, create_words=create_words)

if config['decoding']['type'] == 'Syllable':
    train_epochs, val_epochs = load_and_split_per_file()
else:
    train_epochs, val_epochs = load_and_split_per_file(directory='words')
    

if config['decoding']['train']:
    train(config, train_epochs, val_epochs)