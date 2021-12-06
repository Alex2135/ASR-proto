import os

DATA_DIR = "D:\\ML\\Speech recognition\\NLP_diploma\\uk"  # Path to main directory whole dataset

TRAIN_PATH = os.path.join(DATA_DIR, "train_spec_classifier.csv")  # Path to classifier csv file
TEST_PATH = os.path.join(DATA_DIR, "test_spec_classifier.csv")  # Path to classifier csv file
CLASSIFIER_SPEC_PATH = os.path.join(DATA_DIR, "mel_spectrograms_classifire")  # Path to spectrogramms images for classifier


TRAIN_REC_PATH = os.path.join(DATA_DIR, "train_spec.csv") # Path to ASR csv file
TRAIN_REC_SPEC_PATH = os.path.join(DATA_DIR, "train_spectrograms") # Path to spectrogramms images for ASR


CONFIG = {
    "Notes": "2 encoders 2 decoders",
    "batch_size": {
        "train": 2,
        "test": 64
    },
    "epochs": 10,
    "learning_rate": 1e-4,
    "pretrain": False,
    "n_encoders": 2,
    "n_decoders": 2,
    "save_model": {
        "state": True,
        "path": os.path.join(DATA_DIR, "model_2.pt")
    }
}