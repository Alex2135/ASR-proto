import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import CommandClassifier as Model
from data_processing import ukr_lang_chars_handle
from data_processing import CommonVoiceUkr
from config import *
from torch.optim import AdamW
from tqdm import tqdm
from model import get_cosine_schedule_with_warmup
import pprint


def train(model, train_dataloader, optimizer, device, epoch=1):
    print(f"Training begin")
    model.train()
    ce_criterion = nn.CrossEntropyLoss()
    running_loss = []
    losses_per_phase = []

    for idx, (X, tgt) in tqdm(enumerate(train_dataloader)):
        tgt_text = tgt["text"]
        tgt_class = torch.Tensor([tgt["label"]]).long().to(device)
        tgt_class = F.one_hot(tgt_class, num_classes=5)

        one_hots = ukr_lang_chars_handle.sentences_to_one_hots(tgt_text, 152).to(device)
        X = X.to(device) #

        emb, output = model(X, one_hots)  # (batch, _, n_class, time), (batch, _, time, n_class)
        loss = ce_criterion(output, tgt_class.float()) # output.shape == (N, C) where N - batch, C - number of classes
        loss.backward()
        optimizer.step()
        #scheduler.step()
        running_loss.append(loss.cpu().detach().numpy())
        losses_per_phase.append(loss.cpu().detach().numpy())
        if (idx + 1) % 25 == 0:  # print every 200 mini-batches
            print(f"Epoch: {epoch}, Last loss: {loss.item():.4f}, Loss phase mean: {np.mean(np.array(losses_per_phase)):.4f}")
            losses_per_phase = []
        optimizer.zero_grad()


def val(model, train_dataloader, device):
    model.eval()
    positive = 0
    train_len = len(train_dataloader)

    print("\n")
    print("Evaluation on train dataset")
    with torch.no_grad():
        for idx, (X, tgt) in tqdm(enumerate(train_dataloader)):
            tgt_text = tgt["text"]
            tgt_class = torch.Tensor([tgt["label"]]).long().to(device)
            tgt_class = F.one_hot(tgt_class, num_classes=5)
            one_hots = ukr_lang_chars_handle.sentences_to_one_hots(tgt_text, 152).to(device)
            X = X.to(device)
            emb, output = model(X, one_hots)
            is_right = torch.argmax(output, dim=-1) == tgt_class
            positive += torch.sum(is_right)

    train_accuracy = positive / train_len
    print(f"Accuracy on TRAIN dataset: {train_accuracy*100:.2f}%\n")


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Making dataset and loader
    ds = CommonVoiceUkr(TRAIN_PATH, TRAIN_SPEC_PATH, batch_size=BATCH_SIZE)
    train_dataloader = DataLoader(ds, shuffle=True, batch_size=BATCH_SIZE)
    train_val_dataloader = DataLoader(ds, shuffle=True, batch_size=BATCH_SIZE)
    epochs = CONFIG["epochs"]
    train_len = len(train_dataloader) * epochs

    tgt_n = 152
    model = Model(n_encoders=CONFIG["n_encoders"], n_decoders=CONFIG["n_decoders"], device=device)
    if CONFIG["pretrain"] == True:
        PATH = os.path.join(DATA_DIR, "model_1.pt")
        model = Model(n_encoders=CONFIG["n_encoders"], n_decoders=CONFIG["n_decoders"], device=device)
        model.load_state_dict(torch.load(PATH))

    # Create optimizator
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    save_model = True
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=train_len//5, num_training_steps=train_len)

    for epoch in range(1, epochs + 1):
        print(f"Epoch №{epoch}")
        train(model, train_dataloader, optimizer, device, epoch=epoch)
        val(model, train_val_dataloader, device)
    if save_model:
        import os
        PATH = os.path.join(DATA_DIR, "model_1.pt")
        print(f"Save model to path: '{PATH}'")
        torch.save(model.state_dict(), PATH)
