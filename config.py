import os

DATA_DIR = "D:\\ML\\Speech recognition\\NLP_diploma\\uk"
TRAIN_PATH = os.path.join(DATA_DIR, "classifier.csv")
TRAIN_SPEC_PATH = os.path.join(DATA_DIR, "spectrograms_classifire")

BATCH_SIZE = 2

CONFIG = {
    "batch_size": BATCH_SIZE,
    "epochs": 5,
    "learning_rate": 0.00001,
    "pretrain": False,
    "n_encoders": 2,
    "n_decoders": 2
}