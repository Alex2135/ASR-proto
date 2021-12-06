import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class EncoderInputsProc(nn.Module):
    def __init__(self, d_inputs, d_model, device="cpu"):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(d_inputs, d_model, kernel_size=7, stride=3),
            nn.ReLU()
        ).to(device)

    def forward(self, X):
        out = self.convs(X)
        out = out.transpose(-1, -2).contiguous()
        return out


# YEP
class DecoderInputsProc(nn.Module):
    def __init__(self, d_inputs, d_model, device="cpu"):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(d_inputs, d_model),
            nn.ReLU()
        ).to(device)

    def forward(self, X):
        out = self.lin(X)
        return out


class AbsolutePositionEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def pos_f(self, row, column, emb_dim) -> Tensor:
        func = (np.sin, np.cos)[column % 2]
        w_k = 1 / np.power(10000, 2 * column / emb_dim)
        pe_i_j = func(row * w_k)
        return Tensor([pe_i_j])

    def position_encoding(self, X):
        assert len(X.shape) >= 3, "X shape must have more then 3 dimension"
        b = X.shape[0]
        h = X.shape[-2]
        w = X.shape[-1]
        pe = torch.zeros((b, h, w))
        for k in range(b):
            for i in range(h):
                for j in range(w):
                    pe[k][i][j] = self.pos_f(i, j, h)

        pe = pe.reshape(b, h, w)
        return pe

    def forward(self, x):
        pe = self.position_encoding(x)
        return pe


class LFFN(nn.Module):
    def __init__(self, dim, dim_bn, dim_hid, dropout=0.3):
        """
        Args:
        dim_bn - int,
            bottleneck dimention
        dim - int,
            dim of input data
        dim_hid - int,
            number of hidden units
        """
        super().__init__()
        self.E1 = nn.Linear(in_features=dim, out_features=dim_bn, bias=False)
        self.D1 = nn.Linear(in_features=dim_bn, out_features=dim_hid, bias=False)
        self.swish = Swish()
        self.dropout = nn.Dropout(dropout)
        self.E2 = nn.Linear(in_features=dim_hid, out_features=dim_bn, bias=False)
        self.D2 = nn.Linear(in_features=dim_bn, out_features=dim, bias=False)

    def forward(self, inputs):
        x = self.E1(inputs)
        x = self.D1(x)
        x = self.swish(x)
        x = self.dropout(x)
        x = self.E2(x)
        y = self.D2(x)
        return y


class MHLA2(nn.Module):
    def __init__(self,
                 num_heads,
                 dim_input_q,
                 dim_input_kv,
                 device="cpu",
                 mask=False
                 ):
        """
        Args:

        """
        super().__init__()
        assert dim_input_q % num_heads == 0, "dim_input_q must be devided on num_heads"
        assert dim_input_kv % num_heads == 0, "dim_input_kv must be devided on num_heads"
        dim_q = dim_input_q // num_heads
        dim_k = dim_input_kv // num_heads
        self.device = device
        self.with_mask = mask
        self.W_Q = torch.ones((num_heads, dim_input_q, dim_q), device=device, requires_grad=True)
        self.W_K = torch.ones((num_heads, dim_input_kv, dim_q), device=device, requires_grad=True)
        self.W_V = torch.ones((num_heads, dim_input_kv, dim_q), device=device, requires_grad=True)
        self.W_O = nn.Linear(dim_k * num_heads, dim_k * num_heads, bias=False)
        self.W_Q = nn.Parameter(nn.init.xavier_uniform_(self.W_Q))
        self.W_K = nn.Parameter(nn.init.xavier_uniform_(self.W_K))
        self.W_V = nn.Parameter(nn.init.xavier_uniform_(self.W_V))
        self.d_q = torch.pow(Tensor([dim_q]).to(device), 1 / 4)
        self.d_k = torch.pow(Tensor([dim_k]).to(device), 1 / 4)
        self.softmax_col = nn.Softmax(dim=-2)
        self.softmax_row = nn.Softmax(dim=-1)

    def mask(self, dim: (int, int)) -> Tensor:
        a, b = dim
        mask = torch.ones(b, a)
        mask = torch.triu(mask, diagonal=0)
        mask = torch.log(mask.T)
        return mask.to(self.device)

    def forward(self, x_q, x_k, x_v):
        x_q, x_k, x_v = x_q.unsqueeze(dim=1), x_k.unsqueeze(dim=1), x_v.unsqueeze(dim=1)
        Q = torch.matmul(x_q, self.W_Q)
        K = torch.matmul(x_k, self.W_K)
        V = torch.matmul(x_v, self.W_V)
        if self.with_mask == True:
            Q += self.mask(Q.shape[-2:])
        A = torch.matmul(self.softmax_col(K.transpose(-1, -2).contiguous() / self.d_k), V)
        B = torch.matmul(self.softmax_row(Q / self.d_q), A)
        # print(f"{B.shape}")
        b, h, w, d = B.shape
        B = self.W_O(B.view(b, w, h * d))

        return B


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(chan_in, chan_in, kernel_size=(1, kernel_size), padding=(0, padding))
        self.conv2 = nn.Conv2d(chan_in, chan_out, kernel_size=(1, 1))

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x


class PointWiseConv(nn.Module):
    def __init__(self, chan_in):
        super().__init__()
        self.pw_conv = nn.Conv2d(in_channels=chan_in, out_channels=1, kernel_size=1)

    def forward(self, inputs):
        x = self.pw_conv(inputs)
        return x


class ConvModule(nn.Module):
    def __init__(self, dim_W, dim_bn, dropout=0.3):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim_W)
        self.pw_conv1 = PointWiseConv(chan_in=1)
        self.glu = GLU(dim=-1)
        self.dw_conv1d = DepthWiseConv1d(1, dim_bn, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(dim_bn)
        self.swish = Swish()
        self.pw_conv2 = PointWiseConv(chan_in=dim_bn)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs.unsqueeze(dim=1)

        x = self.ln1(x)

        x = self.pw_conv1(x)
        x = self.glu(x)
        x = self.dw_conv1d(x)

        x = self.bn(x)
        x = self.swish(x)
        x = self.pw_conv2(x)
        x = self.dropout(x)
        x = inputs.squeeze(dim=1)
        return x


class LAC(nn.Module):
    def __init__(self, d_model, n_heads=2, device="cpu", dropout=0.1):
        super().__init__()
        self.lffn1 = LFFN(dim=d_model, dim_bn=256, dim_hid=1024)
        self.do1 = nn.Dropout(dropout)
        self.mhlsa = MHLA2(num_heads=n_heads, dim_input_q=d_model, dim_input_kv=d_model, device=device)
        self.do2 = nn.Dropout(dropout)
        self.conv_module = ConvModule(d_model, dim_bn=8)
        self.do3 = nn.Dropout(dropout)
        self.lffn2 = LFFN(dim=d_model, dim_bn=512, dim_hid=1024)
        self.do4 = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, inputs):
        x = inputs
        x = x + 1 / 2 * self.do1(self.lffn1(x))
        x = x + self.do2(self.mhlsa(x, x, x))
        x = x + self.do3(self.conv_module(x))
        x = x + 1 / 2 * self.do4(self.lffn2(x))
        x = self.ln(x)
        return inputs + x


class Encoder(nn.Module):
    def __init__(self, d_model, n_encoders=2, n_heads=4, device="cpu"):
        super().__init__()
        self.lacs = nn.Sequential(*[LAC(d_model=d_model, n_heads=n_heads, device=device) for i in range(n_encoders)])

    def forward(self, inputs):
        x = self.lacs(inputs)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim_tgt, dim_mem, n_heads=4, device="cpu", dropout=0.1):
        super().__init__()
        self.mhla_with_mask = MHLA2(num_heads=n_heads, dim_input_q=dim_tgt, dim_input_kv=dim_tgt, mask=True, device=device)
        self.do1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(dim_tgt)
        self.mhla_with_memory = MHLA2(num_heads=n_heads, dim_input_q=dim_tgt, dim_input_kv=dim_mem, device=device)
        self.do2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(dim_tgt)
        self.lffn = LFFN(dim=dim_mem, dim_bn=512, dim_hid=1024)
        self.do3 = nn.Dropout(dropout)
        self.ln3 = nn.LayerNorm(dim_tgt)

    def forward(self, mem, y):
        y = y + self.do1(self.mhla_with_mask(y, y, y))
        y = self.ln1(y)
        y = y + self.do2(self.mhla_with_memory(y, mem, mem))
        y = self.ln2(y)
        y = y + self.do3(self.lffn(y))
        y = self.ln3(y)
        return y


class Decoder(nn.Module):
    def __init__(self, d_model, n_decoders=4, n_heads=4, device="cpu"):
        super().__init__()
        self.device = device
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dim_tgt=d_model, n_heads=n_heads, dim_mem=d_model, device=device)
            for _ in range(n_decoders)])

    def forward(self, mem, tgt):
        y = tgt.to(self.device)
        #print(f"{y.shape=}")

        for dec in self.dec_blocks:
            y = dec(mem, y)

        return y


class EfficientConformer(nn.Module):
    def __init__(self,
                 n_encoders=2,
                 n_decoders=2,
                 d_model=64,
                 d_inputs=768,
                 d_outputs=38,
                 n_heads_enc=4,
                 n_heads_dec=4,
                 device="cpu"):
        super().__init__()
        self.enc_proc = EncoderInputsProc(d_inputs=d_inputs, d_model=d_model, device=device)
        self.dec_proc = DecoderInputsProc(d_inputs=d_outputs, d_model=d_model, device=device)
        self.pos_enc_inp = AbsolutePositionEncoding()
        self.pos_enc_out = AbsolutePositionEncoding()
        self.encoder = Encoder(n_encoders=n_encoders, d_model=d_model, n_heads=n_heads_enc, device=device)
        self.decoder = Decoder(n_decoders=n_decoders, d_model=d_model, n_heads=n_heads_dec, device=device)
        self.device = device
        self.to(device)

    def to(self, device, *args, **kwargs):
        self = super().to(device, *args, **kwargs)
        self.device = device
        return self

    def forward(self, inputs, tgt):
        x = self.enc_proc(inputs)
        x = x + self.pos_enc_inp(x).to(self.device)
        x = self.encoder(x)
        y = self.dec_proc(tgt)
        y = y + self.pos_enc_out(y).to(self.device)
        y = self.decoder(x, y)

        return x, y


class EfConfRecognizer(EfficientConformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = kwargs.get("device", "cpu")
        self.lin_out = nn.Sequential(
            nn.Linear(kwargs["d_model"], 38),
            nn.LogSoftmax(dim=-1)
        ).to(self.device)

    def forward(self, inputs, tgt):
        inputs = inputs.to(self.device)
        tgt = tgt.to(self.device)
        emb, out = super().forward(inputs, tgt)
        out = self.lin_out(out)
        return emb, out


class EfConfClassifier(EfficientConformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = kwargs.get("device", "cpu")
        self.lin_out = nn.Sequential(
            nn.Linear(kwargs["d_model"], kwargs["d_outputs"]),
            nn.Softmax(dim=-1)
        ).to(self.device)

    def forward(self, inputs, tgt):
        emb, out = super().forward(inputs, tgt)
        out = torch.sum(out, dim=-2)
        out = self.lin_out(out)
        return emb, out


if __name__ == "__main__":
    d_model = 64

    X1 = torch.randn(1, 206, 768)  # X1 = torch.randn(1, 768, 206)
    X2 = torch.randn(2, 1024, 768)  # X2 = torch.randn(2, 768, 1024)

    Y1 = torch.randn(1, 29, 38)
    Y2 = torch.randn(2, 151, 38)

    conf = EfConfClassifier(n_encoders=2, n_decoders=2, d_model=d_model)
    emb, out = conf(inputs=X2, tgt=Y2)

    print(f"out shape: {out.shape=}")
    print(f"out.shape == Y2.shape :{out.shape==Y2.shape}")