import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import Conformer as con
from data_processing import ukr_lang_chars_handle
from data_processing import CommonVoiceUkr
from config import *
from torch.optim import RAdam
from tqdm import tqdm
import pprint
import numpy as np
import wandb
#wandb.init(project="ASR", entity="alex2135")

#wandb.config = CONFIG

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Making dataset and loader
ds = CommonVoiceUkr(TRAIN_PATH, TRAIN_SPEC_PATH, batch_size=BATCH_SIZE)
train_dataloader = DataLoader(ds, shuffle=True, batch_size=BATCH_SIZE)


# Get model result
tgt_n = 152
src = torch.rand(BATCH_SIZE, 1, 256, 1024).to(device)
tgt = torch.rand(1, 1, 38, 152).to(device)
model = con(device=device)
result = model(src, tgt)


# Create optimizator
optimizer = RAdam(model.parameters())


# Create CTC criterion
b, cnls, t, clss = result.shape
result = result.view(t * cnls, b, clss).to(device)
target = torch.randn(BATCH_SIZE, tgt_n).to(device)
input_lengths = torch.full(size=(BATCH_SIZE,), fill_value=result.shape[0], dtype=torch.long).to(device)
target_lengths = torch.full(size=(BATCH_SIZE,), fill_value=tgt.shape[-1], dtype=torch.long).to(device)
criterion = nn.CTCLoss(blank=ukr_lang_chars_handle.token_to_index["<blank>"], zero_infinity=True)
# loss = criterion(result, target, input_lengths, target_lengths)


running_loss = []
epochs = 5
for epoch in range(1, epochs + 1):
    for idx, (X, tgt) in tqdm(enumerate(train_dataloader)):
        optimizer.zero_grad()

        one_hots = ukr_lang_chars_handle.sentences_to_one_hots(tgt, 152).to(device)
        X = X.to(device)
        output = model(X, one_hots)  # (batch, _, time, n_class)
        b, cnls, t, clss = output.shape
        output = output.view(t * cnls, b, clss)  # (time, batch, n_class)
        output = F.log_softmax(output, dim=2)

        indeces = ukr_lang_chars_handle.sentences_to_indeces(tgt).to(device)
        target_lengths = torch.full(size=(BATCH_SIZE,), fill_value=indeces.shape[-1], dtype=torch.long).to(device)
        loss = criterion(output, indeces, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        running_loss.append(loss)
        #wandb.log({"loss": loss})

        if torch.isnan(loss) or torch.isinf(loss):
            print("Target label:", tgt)
            print("Running loss:")
            pprint.pprint(running_loss)
            print(output.shape)
            print("Is nan in output:", torch.sum(torch.isnan(output)))
            print("Is inf in output:", torch.sum(torch.isinf(output)))
            pprint.pprint(output)
            break
        if (idx + 1) % 200 == 0:  # print every 200 mini-batches
            running_loss = [t.cpu().detach().numpy() if type(t) is torch.Tensor else t for t in running_loss]
            running_loss = np.array(running_loss)
            print(f"Epoch: {epoch}, Last loss: {loss:.4f}, Loss mean: {np.mean(running_loss):.4f}")
            running_loss = list(running_loss)