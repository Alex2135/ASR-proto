import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import wandb
from tqdm import tqdm

from config import *
from data_processing import UkrVoiceDataset
from model import EfConfClassifier as Model

from torchnet.meter import APMeter


def eval_by_dl(model, dl, device, zero_labels=False, wb=None, caption="train"):
    model.eval()
    positive = 0
    train_len = len(dl.dataset)
    apmeter = APMeter()

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

            # if zero_labels:
            #      print(f"\n{A=}\n{tgt_=}")
            apmeter.add(A, tgt_)
            is_right = (A == B)
            positive += torch.sum(is_right)

    accuracy = positive / train_len
    if wb:
        wb.log({
            f"{caption}, accuracy": accuracy
        })
    print(f"Accuracy on {caption.upper()} dataset: {accuracy * 100:.2f}%\n")
    print(f"mAP on {caption.upper()} dataset: {apmeter.value()}%\n")


def eval_sample(inputs: torch.Tensor, model: nn.Module, device="cpu") -> int:
    model.eval()
    result = -1

    with torch.no_grad():
        tgt_class = torch.Tensor(0).long().to(device)
        tgt_class = F.one_hot(tgt_class, num_classes=5).unsqueeze(dim=1).float()
        x = inputs.to(device)  #
        x = x.squeeze(dim=1).permute(0, 2, 1)
        emb, output = model(x, tgt_class)
        result = torch.argmax(output, dim=-1).reshape(-1)

    return result


def collate_fn(data):
    xs, lbls = zip(*data)
    xs_out = pad_sequence([x.permute(0, 2, 1).squeeze(dim=0) for x in xs], batch_first=True)
    lbl1 = lbls[0]
    d_out = {}
    for key in lbl1.keys():
        d_out[key] = [d[key] for d in lbls]
    return xs_out, d_out


def main(model: nn.Module = None, wandb_stat = None):
    #torch.random.manual_seed(42)
    wandb_stat = None #wandb.init(project="ASR", entity="Alex2135", config=CONFIG)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Making dataset and loader
    ds_train = UkrVoiceDataset(TRAIN_PATH, CLASSIFIER_SPEC_PATH,  pad_dim1=103)
    ds_test = UkrVoiceDataset(TEST_PATH, CLASSIFIER_SPEC_PATH, pad_dim1=103)
    train_val_dl = DataLoader(ds_train, shuffle=True, collate_fn=collate_fn, batch_size=CONFIG["batch_size"]["test"])
    test_val_dl = DataLoader(ds_test, shuffle=True, collate_fn=collate_fn, batch_size=CONFIG["batch_size"]["test"])

    model = Model(n_encoders=CONFIG["n_encoders"],
                  n_decoders=CONFIG["n_decoders"],
                  d_inputs=103,
                  d_model=64,
                  d_outputs=5,
                  device=device)
    state_dict = torch.load(CONFIG["save_model"]["path"])
    model.load_state_dict(state_dict)

    with torch.inference_mode(True):
        print("\n")
        print("Evaluation on train dataset with TRUE LABELS")
        eval_by_dl(model, train_val_dl, device, wb=wandb_stat,
                   caption="train dataset with TRUE LABELS")

        print("\n")
        print("Evaluation on test dataset with TRUE LABELS")
        eval_by_dl(model, test_val_dl, device, wb=wandb_stat,
                   caption="test dataset with TRUE LABELS")

        print("\n")
        print("Evaluation on train dataset with ZERO LABELS")
        eval_by_dl(model, train_val_dl, device, zero_labels=True, wb=wandb_stat,
                   caption="train dataset with ZERO LABELS")

        print("\n")
        print("Evaluation on test dataset with ZERO LABELS")
        eval_by_dl(model, test_val_dl, device, zero_labels=True, wb=wandb_stat,
                   caption="test dataset with ZERO LABELS")


if __name__ == "__main__":
    main()
