import os

DATA_DIR = "D:\\ML\\Speech recognition\\NLP_diploma\\uk"
TRAIN_PATH = os.path.join(DATA_DIR, "train_spec.csv")
TRAIN_SPEC_PATH = os.path.join(DATA_DIR, "train_spectrograms")

BATCH_SIZE = 8

CONFIG = {
    "batch_size": BATCH_SIZE,
    "epochs": 10,
    "learning_rate": 0.0001,
    "pretrain": True
}