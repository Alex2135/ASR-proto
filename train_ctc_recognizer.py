import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Conformer as con
from data_processing import ukr_lang_chars_handle
from data_processing import CommonVoiceUkr
from config import *
from torch.optim import RAdam, AdamW
from tqdm import tqdm
import pprint
import numpy as np
#from torch.optim.lr_scheduler import MultiplicativeLR
from model import MaskedSoftmaxCELoss
from model import get_cosine_schedule_with_warmup
import wandb

# wandb.init(project="ASR", entity="alex2135")

# wandb.config = CONFIG

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Making dataset and loader
ds = CommonVoiceUkr(TRAIN_PATH, TRAIN_SPEC_PATH, batch_size=BATCH_SIZE)
train_dataloader = DataLoader(ds, shuffle=True, batch_size=BATCH_SIZE)
train_len = len(train_dataloader) * CONFIG["epochs"]
print("train len:", train_len)

def eleminate_channels(X: torch.Tensor) -> torch.Tensor:
    b, c, h, w = X.shape
    X = X.view(b, c*h, w)
    return X

tgt_n = 152
model = con(n_encoders=CONFIG["n_encoders"], n_decoders=CONFIG["n_decoders"], device=device)
if CONFIG["pretrain"] == True:
    PATH = os.path.join(DATA_DIR, "model_1.pt")
    model = con(n_encoders=CONFIG["n_encoders"], n_decoders=CONFIG["n_decoders"], device=device)
    model.load_state_dict(torch.load(PATH))

# Create optimizator
optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=train_len//5, num_training_steps=train_len)

# Create CTC criterion
alpha_loss = torch.Tensor([0.7]).to(device)
ctc_criterion = nn.CTCLoss(blank=ukr_lang_chars_handle.token_to_index["<blank>"], zero_infinity=True)
ce_criterion = nn.CrossEntropyLoss()

running_loss = []
losses_per_phase = []
epochs = CONFIG["epochs"]
try:
    for epoch in range(1, epochs + 1):
        print(f"Epoch №{epoch}")
        for idx, (X, tgt) in tqdm(enumerate(train_dataloader)):

            tgt_text = tgt["text"]
            tgt_class = tgt["label"]

            one_hots = ukr_lang_chars_handle.sentences_to_one_hots(tgt_text, 152).to(device)
            X = X.to(device) #

            emb, output = model(X, one_hots)  # (batch, _, n_class, time), (batch, _, time, n_class)
            b, cnls, t, clss = output.shape
            output = output.view(t * cnls, b, clss)  # (time, batch, n_class)
            output = F.log_softmax(output, dim=-1)
            indeces = ukr_lang_chars_handle.sentences_to_indeces(tgt_text).to(device)

            input_lengths = torch.full(size=(BATCH_SIZE,), fill_value=t, dtype=torch.long).to(device)
            target_lengths = torch.full(size=(BATCH_SIZE,), fill_value=indeces.shape[-1], dtype=torch.long).to(device)
            ctc_loss = ctc_criterion(output.to(device), indeces, input_lengths, target_lengths)

            print(f"{output.shape=}")
            break
            #print(f"\n{emb.shape=}\n{one_hots.shape=}")
            #emb, one_hots = eleminate_channels(emb.to(device)), eleminate_channels(one_hots.float())
            #ce_loss = ce_criterion(emb, one_hots)

            #loss = alpha_loss * torch.log(ce_loss) + (1-alpha_loss) * torch.log(ctc_loss)
            loss = ctc_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            #print(f"\n{ctc_loss.item()=}, {torch.log(ctc_loss)}\n{ce_loss.item()=}, {torch.log(ce_loss)}")
            running_loss.append(loss.cpu().detach().numpy())
            losses_per_phase.append(loss.cpu().detach().numpy())
            # wandb.log({"loss": loss})

            if torch.isnan(loss) or torch.isinf(loss):
                print("Target label:", tgt)
                print("Running loss:")
                pprint.pprint(running_loss)
                print(output.shape)
                print("Is nan in output:", torch.sum(torch.isnan(output)))
                print("Is inf in output:", torch.sum(torch.isinf(output)))
                pprint.pprint(output)
                break
            if (idx + 1) % 50 == 0:  # print every 200 mini-batches
                print(f"Epoch: {epoch}, Last loss: {loss.item():.4f}, Loss phase mean: {np.mean(np.array(losses_per_phase)):.4f}")
                losses_per_phase = []
            optimizer.zero_grad()
    import os
    PATH = os.path.join(DATA_DIR, "model_1.pt")
    print(PATH)
    torch.save(model.state_dict(), PATH)
except Exception as e:
    print(e)
