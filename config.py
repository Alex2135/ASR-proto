import os

DATA_DIR = "D:\\ML\\Speech recognition\\NLP_diploma\\uk" # Path to main directory whole dataset
TRAIN_PATH = os.path.join(DATA_DIR, "classifier.csv") # Path to classifier csv file
TRAIN_SPEC_PATH = os.path.join(DATA_DIR, "spectrograms_classifire") # Path to spectrogramms images for classifier
TRAIN_REC_PATH = os.path.join(DATA_DIR, "train_spec.csv") # Path to ASR csv file
TRAIN_REC_SPEC_PATH = os.path.join(DATA_DIR, "train_spectrograms") # Path to spectrogramms images for ASR

BATCH_SIZE = 32

CONFIG = {
    "Notes": "3 encoders 1 decoders",
    "batch_size": BATCH_SIZE,
    "epochs": 10, # 82, 55
    "learning_rate": 1e-6, # max: 1e-6, min: 1e-7, begin: 5e-7
    "pretrain": False,
    "n_encoders": 3,
    "n_decoders": 1,
    "save_model": True
}