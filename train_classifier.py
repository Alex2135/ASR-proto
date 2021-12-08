import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import wandb
import numpy as np
from tqdm import tqdm
import pprint

from config import *
from data_processing import ukr_lang_chars_handle
from data_processing import UkrVoiceDataset
from model import EfConfClassifier as Model
from model import get_cosine_schedule_with_warmup, OneCycleLR

import os
from copy import deepcopy


def train(model, train_dataloader, optimizer, device, scheduler=None, epoch=1, wb=None):
    print(f"Training begin")
    model.train()
    ce_criterion = nn.CrossEntropyLoss()
    running_loss = []
    losses_per_phase = []
    train_len = len(train_dataloader)

    for idx, (X, tgt) in tqdm(enumerate(train_dataloader)):
        tgt_lbls = torch.Tensor(tgt["label"]).long().to(device)
        mask = torch.rand(tgt_lbls.shape) > CONFIG["dropout_inputs"]  # creating mask with threshold
        mask = mask.long().to(device)
        tgt_lbls_do = mask * tgt_lbls

        tgt_class = F.one_hot(tgt_lbls, num_classes=5).unsqueeze(dim=1).float()
        tgt_class_do = F.one_hot(tgt_lbls_do, num_classes=5).unsqueeze(dim=1).float()
        tgt_class_do = tgt_class_do.to(device)

        X = X.to(device)  #
        X = X.squeeze(dim=1).permute(0, 2, 1)

        emb, output = model(X, tgt_class_do)  # X: (batch, time, n_class), tgt_class_do: (batch, time, n_class)
        tgt_class = tgt_class.squeeze(dim=1).float()
        loss = ce_criterion(output, tgt_class.squeeze(dim=1).float())
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
        if (idx + 1) % (train_len // 10) == 0:  # print every 200 mini-batches
            loss_mean = np.mean(np.array(losses_per_phase))
            print(f"Epoch: {epoch}, Last loss: {loss.item():.4f}, Loss phase mean: {loss_mean:.4f}")
            if wb:
                wb.log({
                    "loss phase mean": loss_mean,
                    "epoch": epoch
                })
            losses_per_phase = []
        optimizer.zero_grad()


def val(model, dl, device, zero_labels=False, wb=None, caption="train", epoch=1):
    model.eval()

    positive = 0
    train_len = len(dl.dataset)

    with torch.no_grad():
        for idx, (X, tgt) in tqdm(enumerate(dl)):
            tgt_ = torch.Tensor(tgt["label"]).long().to(device)
            tgt_class = torch.zeros_like(tgt_) if zero_labels else tgt_
            tgt_ = F.one_hot(tgt_, num_classes=5).long()
            tgt_class = F.one_hot(tgt_class, num_classes=5).unsqueeze(dim=1).float()
            X = X.to(device)  #
            X = X.squeeze(dim=1).permute(0, 2, 1)
            emb, output = model(X, tgt_class)

            A = torch.argmax(output, dim=-1).reshape(-1)
            B = torch.argmax(tgt_, dim=-1).reshape(-1)

            is_right = (A == B)
            positive += torch.sum(is_right)

    accuracy = positive / train_len
    postfix = "(zero)" if zero_labels else ""
    if wb:
        wb.log({
            f"{caption}, accuracy {postfix}": accuracy,
            "epoch": epoch,
        })
    print(f"Accuracy on {caption.upper()} dataset: {accuracy * 100:.2f}%\n")


def get_scheduler(epochs, train_len, optimizer, scheduler_name="cosine_with_warmup", wb=None):
    if wb:
        wb.config["scheduler"] = scheduler_name
    if scheduler_name == "cosine_with_warmup":
        return get_cosine_schedule_with_warmup(optimizer,
                                               num_warmup_steps=epochs // 5,
                                               num_training_steps=epochs - epochs // 5,
                                               num_cycles=1.25
                                               )
    elif scheduler_name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer)
    elif scheduler_name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    elif scheduler_name == "one_circle":
        return OneCycleLR(optimizer,
                          max_lr=CONFIG["learning_rate"] * 10,
                          total_steps=train_len)


def collate_fn(data):
    xs, lbls = zip(*data)
    xs_out = pad_sequence([x.permute(0, 2, 1).squeeze(dim=0) for x in xs], batch_first=True)
    lbl1 = lbls[0]
    d_out = {}
    for key in lbl1.keys():
        d_out[key] = [d[key] for d in lbls]
    return xs_out, d_out


def main():
    # torch.random.manual_seed(42)
    wandb_stat = wandb.init(project="ASR", entity="Alex2135", config=CONFIG)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Making dataset and loader
    ds_train = UkrVoiceDataset(TRAIN_PATH, CLASSIFIER_SPEC_PATH, pad_dim1=103)
    ds_test = UkrVoiceDataset(TEST_PATH, CLASSIFIER_SPEC_PATH, pad_dim1=103)
    train_dl = DataLoader(ds_train, shuffle=True, collate_fn=collate_fn, batch_size=CONFIG["batch_size"]["train"])
    train_val_dl = DataLoader(ds_train, shuffle=True, collate_fn=collate_fn, batch_size=CONFIG["batch_size"]["test"])
    test_val_dl = DataLoader(ds_test, shuffle=True, collate_fn=collate_fn, batch_size=CONFIG["batch_size"]["test"])

    # Set model
    epochs = CONFIG["epochs"]
    train_len = len(train_dl) * epochs
    model = Model(n_encoders=CONFIG["n_encoders"],
                  n_decoders=CONFIG["n_decoders"],
                  d_inputs=103,
                  d_model=64,
                  d_outputs=5,
                  device=device)
    if CONFIG["pretrain"]:
        PATH = CONFIG["save_model"]["path"]
        model.load_state_dict(torch.load(PATH))

    # Create optimizator
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = get_scheduler(CONFIG["epochs"], train_len, optimizer, scheduler_name="constant",
                              wb=wandb_stat)

    for epoch in range(1, epochs + 1):
        print(f"Epoch №{epoch}")
        train(model, train_dl, optimizer, device, scheduler=scheduler, epoch=epoch, wb=wandb_stat)

        print("\n")
        print("Evaluation on TRAIN dataset WITH ZEROS")
        val(model, train_val_dl, device, wb=wandb_stat, caption="train", zero_labels=True, epoch=epoch)

        print("\n")
        print("Evaluation on TEST dataset WITH ZEROS")
        val(model, test_val_dl, device, wb=wandb_stat, caption="test", zero_labels=True, epoch=epoch)

        scheduler.step(epoch)
        last_lr = float(scheduler.get_last_lr()[0])
        print(f"scheduler last_lr: {last_lr}")
        if wandb_stat:
            wandb_stat.log({"scheduler lr": last_lr, "epoch": epoch})

    if CONFIG["save_model"]["state"]:
        PATH = CONFIG["save_model"]["path"]
        print(f"Save model to path: '{PATH}'")
        state_dict = deepcopy(model.state_dict())
        torch.save(state_dict, PATH )


if __name__ == "__main__":
    main()
