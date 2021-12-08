import os

DATA_DIR = "D:\\ML\\Speech recognition\\NLP_diploma\\uk"  # Path to main directory whole dataset

TRAIN_PATH = os.path.join(DATA_DIR, "train_spec_classifier.csv")  # Path to classifier csv file
TEST_PATH = os.path.join(DATA_DIR, "test_spec_classifier.csv")  # Path to classifier csv file
CLASSIFIER_SPEC_PATH = os.path.join(DATA_DIR, "mel_spectrograms_classifire")  # Path to spectrogramms images for classifier


TRAIN_REC_PATH = os.path.join(DATA_DIR, "train_spec_rec.csv") # Path to ASR csv file
TRAIN_REC_SPEC_PATH = os.path.join(DATA_DIR, "train_spectrograms") # Path to spectrogramms images for ASR


CONFIG = {
    "Notes": "Lower LR + cosine warmup. 3 encoders 3 decoders",
    "batch_size": {
        "train": 1,
        "test": 96
    },
    "epochs": 20,
    "learning_rate": 1e-6,
    "pretrain": True,
    "n_encoders": 3,
    "n_decoders": 3,
    "dropout_inputs": 1,
    "save_model": {
        "state": True,
        "path": os.path.join(DATA_DIR, "model_3_3.pt")
    }
}