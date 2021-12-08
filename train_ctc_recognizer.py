import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import wandb
import numpy as np
from tqdm import tqdm
from pprint import pprint

from config import *
from data_processing import ukr_lang_chars_handle
from data_processing import UkrVoiceDataset
from model import EfConfRecognizer as Model
from model import get_cosine_schedule_with_warmup, OneCycleLR

import os
from copy import deepcopy


def train(model, train_dataloader, optimizer, device, scheduler=None, epoch=1, wb=None):
    print(f"Training begin")
    model.train()
    ctc_criterion = nn.CTCLoss(reduction="none", blank=ukr_lang_chars_handle.token_to_index["<blank>"],
                               zero_infinity=True)
    running_loss = []
    losses_per_phase = []
    train_len = len(train_dataloader)

    for idx, (X, tgt) in tqdm(enumerate(train_dataloader)):
        tgt_text = tgt["text"]

        tgt_lengths = [len(txt) for txt in tgt_text]
        tgt_max_len = max(tgt_lengths)
        one_hots = ukr_lang_chars_handle.sentences_to_one_hots(tgt_text, tgt_max_len).to(device)
        one_hots = one_hots.squeeze(dim=1).permute(0, 2, 1).float()

        X = X.to(device)  #
        X = X.squeeze(dim=1).permute(0, 2, 1)
        # print(f"{X.shape=}")
        # print(f"{one_hots.shape=}")

        emb, output = model(X, one_hots)  # (batch, time, n_class), (batch, time, n_class)
        output = output.permute(1, 0, 2).to(device).detach().requires_grad_()
        indeces = ukr_lang_chars_handle.sentences_to_indeces(tgt_text).to(device)

        """
        print("tgt_text:")
        pprint(tgt_text)
        print("tgt_len")
        pprint(tgt_lengths)
        print("indeces:")
        pprint(indeces)
        print(f"Inputs shape: {output.shape}")
        print(f"Tgt shape: {indeces.shape}")
        print(f"one_hots shape: {one_hots.shape}")
        """

        input_lengths = torch.full(size=(output.shape[1],), fill_value=output.shape[0], dtype=torch.long).to(device)
        target_lengths = torch.Tensor(tgt_lengths).long().to(device)

        loss = ctc_criterion(output, indeces, input_lengths, target_lengths)
        # print(f"{loss=}")
        if wb:
            wb.log({
                "loss": loss.item(),
                "epoch": epoch
            })
        # print(f"{loss=}")
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
            tgt_text = " "  #  tgt["text"]
            tgt_class = torch.Tensor(tgt["label"]).long().to(device)
            tgt_class = F.one_hot(tgt_class, num_classes=5)
            one_hots = ukr_lang_chars_handle.sentences_to_one_hots(tgt_text, 152).to(device)
            one_hots = one_hots.squeeze(dim=1).permute(0, 2, 1).float()
            X = X.to(device)  #
            X = X.squeeze(dim=1).permute(0, 2, 1)
            emb, output = model(X, one_hots)
            #print(f"\n{output.shape=}\n{one_hots.shape=}")
            A = torch.argmax(output, dim=-1)
            B = torch.argmax(one_hots, dim=-1)
            is_right = (A == B)
            positive += torch.sum(is_right)

    train_accuracy = positive / train_len
    if wb:
        wb.log({
            "train accuracy": train_accuracy,
            "epoch": epoch
        })
    print(f"Accuracy on TRAIN dataset: {train_accuracy*100:.2f}%\n")


def get_scheduler(epochs, train_len, optimizer, scheduler_name="cosine_with_warmup", wb=None):
    if wb:
        wb.config["scheduler"] = scheduler_name
    if scheduler_name == "cosine_with_warmup":
        return get_cosine_schedule_with_warmup(optimizer,
                                               num_warmup_steps=epochs//5,
                                               num_training_steps=epochs - epochs//5,
                                               num_cycles=0.5)
    elif scheduler_name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer)
    elif scheduler_name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    elif scheduler_name == "one_circle":
        return OneCycleLR(optimizer,
                          max_lr=CONFIG["learning_rate"]*10,
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
    wandb_stat = None  # wandb.init(project="ASR", entity="Alex2135", config=CONFIG)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Making dataset and loader
    ds_train = UkrVoiceDataset(TRAIN_PATH, CLASSIFIER_SPEC_PATH, pad_dim1=103)
    ds_test = UkrVoiceDataset(TEST_PATH, CLASSIFIER_SPEC_PATH, pad_dim1=103)
    train_dl = DataLoader(ds_train, shuffle=True, collate_fn=collate_fn, batch_size=CONFIG["batch_size"]["train"])
    train_val_dl = DataLoader(ds_train, shuffle=True, collate_fn=collate_fn, batch_size=CONFIG["batch_size"]["test"])
    test_val_dl = DataLoader(ds_test, shuffle=True, collate_fn=collate_fn, batch_size=CONFIG["batch_size"]["test"])

    epochs = CONFIG["epochs"]
    train_len = len(train_dl) * epochs

    model = Model(n_encoders=CONFIG["n_encoders"],
                  n_decoders=CONFIG["n_decoders"],
                  d_inputs=103,
                  d_model=64,
                  d_outputs=38,
                  device=device)

    # Create optimizator
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    save_model = False
    scheduler = get_scheduler(CONFIG["epochs"], train_len, optimizer, scheduler_name="constant", wb=wandb_stat)

    for epoch in range(1, epochs + 1):
        print(f"Epoch №{epoch}")
        train(model, train_dl, optimizer, device, scheduler=scheduler, epoch=epoch, wb=wandb_stat)
        val(model, train_dl, device, epoch, wb=wandb_stat)
        scheduler.step(epoch)
        print(f"scheduler last_lr: {scheduler.get_last_lr()[0]}")
        if wandb_stat:
            wandb_stat.log({"scheduler lr": scheduler.get_last_lr()[0]})

    if CONFIG["save_model"]["state"]:
        PATH = CONFIG["save_model"]["path"]
        print(f"Save model to path: '{PATH}'")
        state_dict = deepcopy(model.state_dict())
        torch.save(state_dict, PATH)


if __name__ == "__main__":
    main()
