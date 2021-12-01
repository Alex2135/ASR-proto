import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

import wandb
import numpy as np
from tqdm import tqdm
import pprint

from config import *
from data_processing import ukr_lang_chars_handle
from data_processing import CommonVoiceUkr
from model import CommandClassifier as Model
from model import get_cosine_schedule_with_warmup, OneCycleLR

import os


def train(model, train_dataloader, optimizer, device, scheduler=None, epoch=1, wb=None):
    print(f"Training begin")
    model.train()
    ce_criterion = nn.CrossEntropyLoss()
    running_loss = []
    losses_per_phase = []

    for idx, (X, tgt) in tqdm(enumerate(train_dataloader)):
        tgt_text = " "#tgt["text"]
        tgt_class = tgt["label"].long().to(device)
        tgt_class = F.one_hot(tgt_class, num_classes=5)

        one_hots = ukr_lang_chars_handle.sentences_to_one_hots(tgt_text, 152).to(device)
        X = X.to(device) #

        emb, output = model(X, one_hots)  # (batch, _, n_class, time), (batch, _, time, n_class)
        loss = ce_criterion(output, tgt_class.float()) # output.shape == (N, C) where N - batch, C - number of classes
        if wb:
            wb.log({
                "loss": loss.item(),
                "epoch": epoch
            })
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        running_loss.append(loss.cpu().detach().numpy())
        losses_per_phase.append(loss.cpu().detach().numpy())
        if (idx + 1) % 25 == 0:  # print every 200 mini-batches
            loss_mean = np.mean(np.array(losses_per_phase))
            print(f"Epoch: {epoch}, Last loss: {loss.item():.4f}, Loss phase mean: {loss_mean:.4f}")
            if wb:
                wb.log({"loss phase mean": loss_mean})
            losses_per_phase = []
        optimizer.zero_grad()


def val(model, train_dataloader, device, epoch, wb=None):
    model.eval()
    positive = 0
    train_len = train_dataloader.sampler.num_samples

    print("\n")
    print("Evaluation on train dataset")
    with torch.no_grad():
        for idx, (X, tgt) in tqdm(enumerate(train_dataloader)):
            tgt_text = " "#tgt["text"]
            tgt_class = tgt["label"].long().to(device)
            tgt_class = F.one_hot(tgt_class, num_classes=5)
            one_hots = ukr_lang_chars_handle.sentences_to_one_hots(tgt_text, 152).to(device)
            X = X.to(device)
            emb, output = model(X, one_hots)
            A = torch.argmax(output, dim=-1)
            B = torch.argmax(tgt_class, dim=-1)
            is_right = (A == B)
            positive += torch.sum(is_right)

    train_accuracy = positive / train_len
    if wb:
        wb.log({
            "train accuracy": train_accuracy,
            "epoch": epoch
        })
    print(f"Accuracy on TRAIN dataset: {train_accuracy*100:.2f}%\n")


def get_scheduler(epochs, train_len, optimizer, scheduler_name="cosine_with_warmup"):
    wandb.config["scheduler"] = scheduler_name
    if scheduler_name == "cosine_with_warmup":
        return get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=epochs//5,
                                                num_training_steps=epochs - epochs//5)
    elif scheduler_name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer)
    elif scheduler_name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    elif scheduler_name == "one_circle":
        return OneCycleLR(optimizer,
                          max_lr=CONFIG["learning_rate"]*10,
                          total_steps=train_len)


def main():
    wandb_stat = wandb.init(project="ASR", entity="Alex2135", config=CONFIG)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Making dataset and loader
    ds = CommonVoiceUkr(TRAIN_PATH, TRAIN_SPEC_PATH, batch_size=BATCH_SIZE)
    train_dataloader = DataLoader(ds, shuffle=True, batch_size=BATCH_SIZE)
    train_val_dataloader = DataLoader(ds, shuffle=True, batch_size=64)
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
    scheduler = get_scheduler(CONFIG["epochs"], train_len, optimizer, "cosine_with_warmup")

    for epoch in range(1, epochs + 1):
        print(f"Epoch №{epoch}")
        train(model, train_dataloader, optimizer, device, scheduler=scheduler, epoch=epoch, wb=wandb_stat)
        val(model, train_val_dataloader, device, epoch, wb=wandb_stat)
        scheduler.step(epoch)
        wandb.log({"scheduler lr": scheduler.get_last_lr()[0]})

    if save_model:
        PATH = os.path.join(DATA_DIR, "model_1.pt")
        print(f"Save model to path: '{PATH}'")
        torch.save(model.state_dict(), PATH)


if __name__ == "__main__":
    main()