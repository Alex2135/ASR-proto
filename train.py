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

# Making dataset and loader
ds = CommonVoiceUkr(TRAIN_PATH, TRAIN_SPEC_PATH)
train_dataloader = DataLoader(ds, shuffle=True, batch_size=BATCH_SIZE)

# Get model result
tgt_n = 152
src = torch.rand(BATCH_SIZE, 1, 256, 1024)
tgt = torch.rand(1, 1, 38, 152)
model = con()
result = model(src, tgt)
for param in model.parameters():
    if param.requires_grad == False:
        print(param)

# Create optimizator
optimizer = RAdam(model.parameters())

# Create CTC criterion
b, cnls, t, clss = result.shape
result = result.view(t * cnls, b, clss)
target = torch.randn(BATCH_SIZE, tgt_n)
input_lengths = torch.full(size=(BATCH_SIZE,), fill_value=result.shape[0], dtype=torch.long)
target_lengths = torch.full(size=(BATCH_SIZE,), fill_value=tgt.shape[-1], dtype=torch.long)
criterion = nn.CTCLoss(zero_infinity=True)
loss = criterion(result, target, input_lengths, target_lengths)

DEBUG_MSGS = False
running_loss = []
epoch = 0
for idx, (X, tgt) in tqdm(enumerate(train_dataloader)):
    optimizer.zero_grad()

    one_hots = ukr_lang_chars_handle.sentences_to_one_hots(tgt, 152)
    b, l, d = one_hots.shape
    one_hots = one_hots.view(b, 1, d, l)
    output = model(X, one_hots) # (batch, _, time, n_class)
    b, cnls, t, clss = output.shape
    output = output.view(t * cnls, b, clss) # (time, batch, n_class)
    output = F.log_softmax(output, dim=2)

    indeces = ukr_lang_chars_handle.sentences_to_indeces(tgt, 152)
    loss = criterion(output, indeces, input_lengths, target_lengths)
    #loss = torch.nan_to_num(output, posinf = 10, nan = 10)
    loss.backward()
    optimizer.step()

    running_loss.append(loss)

    #print("Loss:", loss)
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
        running_loss = [t.detach().numpy() if type(t) is torch.Tensor else t for t in running_loss]
        running_loss = np.array(running_loss)
        zeros = np.sum(running_loss == 0)
        print(f"Loss mean: {np.mean(running_loss)}")
        print(f"Zeros percent { round(zeros / len(running_loss) * 100, 4)}%")
        wthout_zeros = running_loss[running_loss != 0]
        print(f"Loss mean without zeros: {np.mean(wthout_zeros)}")
        running_loss = list(running_loss)