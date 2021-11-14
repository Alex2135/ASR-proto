import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import Conformer as con
from data_processing import ukr_lang_chars_handle
from data_processing import CommonVoiceUkr
from config import *

src = torch.rand(1, 1, 256, 1024)
tgt = torch.rand(1, 1, 38, 152)


model = con()
result = model(src, tgt)
result = result.transpose(-1, -2)
print(result.shape)

sent = "Московитам дозволено створити свою державу а татарам чеченцям – ні Але це – расизм"
sent_to_idxs = ukr_lang_chars_handle.sentence_to_one_hots(sent)
print("Sentence:", sent)
print("Sentence to indeces:", sent_to_idxs)
idxs_to_sent = ukr_lang_chars_handle.one_hots_to_sentence(sent_to_idxs)
print("Indeces to sentence:", idxs_to_sent)

ds = CommonVoiceUkr(TRAIN_PATH, TRAIN_SPEC_PATH)
train_dataloader = DataLoader(ds, shuffle=True, batch_size=1)

for X, tgt in train_dataloader:
    print(X.shape, tgt)

    one_hots = ukr_lang_chars_handle.sentences_to_one_hots(tgt)
    print(one_hots.shape)

    # y = model(X, tgt)
    break
