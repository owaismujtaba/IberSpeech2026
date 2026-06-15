import os

from src.utils.config_parser import load_config

from src.data.create_words_epochs import create_words_epochs
from src.data.create_syllable_epochs import create_syllable_epochs


config = load_config()

if config['workflow']['create_syllable_epochs']:
    create_syllable_epochs()
if config['workflow']['create_words_epochs']:
    create_words_epochs()


