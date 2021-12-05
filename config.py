import os

DATA_DIR = "D:\\ML\\Speech recognition\\NLP_diploma\\uk"  # Path to main directory whole dataset

TRAIN_PATH = os.path.join(DATA_DIR, "train_spec_classifier.csv")  # Path to classifier csv file
TRAIN_SPEC_PATH = os.path.join(DATA_DIR, "mel_spectrograms_classifire")  # Path to spectrogramms images for classifier
TEST_PATH = os.path.join(DATA_DIR, "test_spec_classifier.csv")  # Path to classifier csv file
TEST_SPEC_PATH = os.path.join(DATA_DIR, "spectrograms_classifire")  # Path to spectrogramms images for classifier


TRAIN_REC_PATH = os.path.join(DATA_DIR, "train_spec.csv") # Path to ASR csv file
TRAIN_REC_SPEC_PATH = os.path.join(DATA_DIR, "train_spectrograms") # Path to spectrogramms images for ASR


BATCH_SIZE = 2

CONFIG = {
    "Notes": "Using mel specs and decrease lr to 1e-4. 2 encoders 2 decoders",
    "batch_size": BATCH_SIZE,
    "epochs": 10, # 82, 55
    "learning_rate": 1e-4,
    "pretrain": False,
    "n_encoders": 2,
    "n_decoders": 2,
    "save_model": True
}