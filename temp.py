import torch
import torch.nn.functional as F
from data_processing import ukr_lang_chars_handle
from config import *
from model import Conformer as con
from data_processing import CommonVoiceUkr
from torch.utils.data import DataLoader
import pprint

device = "cpu"

PATH = os.path.join(DATA_DIR, "model_1.pt")
model = con(n_encoders=8, n_decoders=8, device=device)
model.load_state_dict(torch.load(PATH))

model.eval()
ds = CommonVoiceUkr(TRAIN_PATH, TRAIN_SPEC_PATH)
train_dataloader = DataLoader(ds, shuffle=True, batch_size=1)

with torch.no_grad():
    X, tgt = next(iter(train_dataloader))
    X = X.to(device)
    print("Target:", tgt)
    print("X shape:", X.shape)
    #tgt = ("",)

    tgt_one_hots = ukr_lang_chars_handle.sentences_to_one_hots(tgt, 152)
    print("tgt to one_hots shape:", tgt_one_hots.shape)
    print("tgt to one_hots:", ukr_lang_chars_handle.one_hots_to_sentences(tgt_one_hots))

    out_data = model(X, tgt_one_hots.to(device))
    out_data = F.softmax(out_data, dim=-1)
    out_data = out_data.cpu()
    print("\n\nOutput data shape:", out_data.shape)
    print("output:", out_data)
    out_data = out_data.transpose(-1, -2).contiguous()
    result = ukr_lang_chars_handle.one_hots_to_sentences(out_data)
    pprint.pprint(len(result))
    pprint.pprint(result)