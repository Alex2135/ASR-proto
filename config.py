import os

DATA_DIR = "D:\\ML\\Speech recognition\\NLP_diploma\\uk"
TRAIN_PATH = os.path.join(DATA_DIR, "classifier.csv")
TRAIN_SPEC_PATH = os.path.join(DATA_DIR, "spectrograms_classifire")

BATCH_SIZE = 2

CONFIG = {
    "Notes": "4 encoders 2 decoders",
    "batch_size": BATCH_SIZE,
    "epochs": 1, # 82, 55
    "learning_rate": 0.0001,
    "pretrain": False,
    "n_encoders": 4,
    "n_decoders": 2
}