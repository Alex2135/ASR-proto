import os

DATA_DIR = "D:\\ML\\Speech recognition\\NLP_diploma\\uk"
TRAIN_PATH = os.path.join(DATA_DIR, "train_spec.csv")
TRAIN_SPEC_PATH = os.path.join(DATA_DIR, "train_spectrograms")

BATCH_SIZE = 4

CONFIG = {
    "batch_size": BATCH_SIZE,
    "epochs": 5,
    "learning_rate": 0.001,

}