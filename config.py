import os

DATA_DIR = "D:\\ML\\Speech recognition\\NLP_diploma\\uk"
TRAIN_PATH = os.path.join(DATA_DIR, "classifier.csv")
TRAIN_SPEC_PATH = os.path.join(DATA_DIR, "spectrograms_classifire")

BATCH_SIZE = 16

CONFIG = {
    "Notes": "3 encoders 1 decoders",
    "batch_size": BATCH_SIZE,
    "epochs": 10, # 82, 55
    "learning_rate": 1e-5, # max: 1e-6, min: 1e-7, begin: 5e-7
    "pretrain": False,
    "n_encoders": 3,
    "n_decoders": 1,
    "save_model": True
}