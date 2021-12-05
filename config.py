import os

DATA_DIR = "D:\\ML\\Speech recognition\\NLP_diploma\\uk"  # Path to main directory whole dataset
TRAIN_PATH = os.path.join(DATA_DIR, "train_spec_classifier.csv")  # Path to classifier csv file
TRAIN_SPEC_PATH = os.path.join(DATA_DIR, "spectrograms_classifire")  # Path to spectrogramms images for classifier
TEST_PATH = os.path.join(DATA_DIR, "test_spec_classifier.csv")  # Path to classifier csv file
TEST_SPEC_PATH = os.path.join(DATA_DIR, "spectrograms_classifire")  # Path to spectrogramms images for classifier

TRAIN_REC_PATH = os.path.join(DATA_DIR, "train_spec.csv") # Path to ASR csv file
TRAIN_REC_SPEC_PATH = os.path.join(DATA_DIR, "train_spectrograms") # Path to spectrogramms images for ASR

BATCH_SIZE = 4

CONFIG = {
    "Notes": "3 encoders 1 decoders",
    "batch_size": BATCH_SIZE,
    "epochs": 10, # 82, 55
    "learning_rate": 1e-3,
    "pretrain": False,
    "n_encoders": 3,
    "n_decoders": 1,
    "save_model": False
}