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
from model import get_cosine_schedule_with_warmup
import wandb

# wandb.init(project="ASR", entity="alex2135")

# wandb.config = CONFIG

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Making dataset and loader
ds = CommonVoiceUkr(TRAIN_PATH, TRAIN_SPEC_PATH, batch_size=BATCH_SIZE)
train_dataloader = DataLoader(ds, shuffle=True, batch_size=BATCH_SIZE)

tgt_n = 152
model = con(n_encoders=8, n_decoders=8, device=device)
if CONFIG["pretrain"] == True:
    PATH = os.path.join(DATA_DIR, "model_1.pt")
    model = con(n_encoders=8, n_decoders=8, device=device)
    model.load_state_dict(torch.load(PATH))

# Create optimizator
optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=3, num_training_steps=CONFIG["epochs"])

# Create CTC criterion
criterion = nn.CTCLoss(blank=ukr_lang_chars_handle.token_to_index["<blank>"], zero_infinity=True)

running_loss = []
epochs = CONFIG["epochs"]
for epoch in range(1, epochs + 1):
    print(f"Epoch №{epoch}")
    for idx, (X, tgt) in tqdm(enumerate(train_dataloader)):
        optimizer.zero_grad()

        one_hots = ukr_lang_chars_handle.sentences_to_one_hots(tgt, 152).to(device)
        X = X.to(device)

        output = model(X, one_hots)  # (batch, _, time, n_class)
        b, cnls, t, clss = output.shape
        output = output.view(t * cnls, b, clss)  # (time, batch, n_class)
        output = F.log_softmax(output, dim=2)
        indeces = ukr_lang_chars_handle.sentences_to_indeces(tgt).to(device)

        input_lengths = torch.full(size=(BATCH_SIZE,), fill_value=t, dtype=torch.long).to(device)
        target_lengths = torch.full(size=(BATCH_SIZE,), fill_value=indeces.shape[-1], dtype=torch.long).to(device)
        loss = criterion(output, indeces.cpu(), input_lengths.cpu(), target_lengths.cpu())
        loss.backward()
        optimizer.step()

        running_loss.append(loss)
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
            running_loss = [t.cpu().detach().numpy() if type(t) is torch.Tensor else t for t in running_loss]
            running_loss = np.array(running_loss)
            print(f"Epoch: {epoch}, Last loss: {loss:.4f}, Loss mean: {np.mean(running_loss):.4f}")
            running_loss = list(running_loss)
    scheduler.step()


import os
PATH = os.path.join(DATA_DIR, "model_1.pt")
print(PATH)
torch.save(model.state_dict(), PATH)