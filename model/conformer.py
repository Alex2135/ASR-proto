import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

    
class AbsolutePositionEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        
    def pos_f(self, row, column, emb_dim):
        func = (np.sin, np.cos)[row % 2]
        w_k = 1/np.power(10000, 2*row/emb_dim)
        pe_i_j = func(column * w_k)
        return torch.Tensor([pe_i_j])
    
    def position_encoding(self, X):
        b, _, h, w = X.shape
        pe = torch.zeros((b, h, w))
        for k in range(b):
            for i in range(h):
                for j in range(w):
                    pe[k][i][j] = self.pos_f(i, j, h)
                
        pe = pe.reshape(b, 1, h, w)
        return pe
    
    def forward(self, x):
        PE = self.position_encoding(x)
        return PE
                
                
class LFFN(nn.Module):
    def __init__(self, inputs_dim, dim_hid):
        """
        Args:
        inputs_dim - tuple, 
            (N, C, H, W) of inputs
        dim_hid - int, 
            number of hidden units
        """
        super().__init__()
        _, _, dim_bn, dim = inputs_dim
        self.E1 = nn.Linear(in_features=dim, out_features=dim_bn, bias=False)
        self.D1 = nn.Linear(in_features=dim_bn, out_features=dim_hid, bias=False)
        self.swish = Swish()
        self.dropout = nn.Dropout(0.5)
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
                 dim_q=16,
                 dim_k=16,
                 device="cpu",
                 mask=False
                 ):
        """
        Args:

        """
        super().__init__()
        self.device=device
        self.with_mask = mask
        self.W_Q = torch.ones((num_heads, dim_input_q, dim_q), device=device, requires_grad=True)
        self.W_K = torch.ones((num_heads, dim_input_kv, dim_q), device=device, requires_grad=True)
        self.W_V = torch.ones((num_heads, dim_input_kv, dim_q), device=device, requires_grad=True)
        self.W_O = nn.Linear(dim_k * num_heads, dim_k * num_heads, bias=False)
        self.W_Q = nn.init.xavier_uniform_(self.W_Q)
        self.W_K = nn.init.xavier_uniform_(self.W_K)
        self.W_V = nn.init.xavier_uniform_(self.W_V)
        self.d_q = torch.pow(torch.Tensor([dim_q]).to(device), 1 / 4)
        self.d_k = torch.pow(torch.Tensor([dim_k]).to(device), 1 / 4)
        self.softmax_col = nn.Softmax(dim=-1)
        self.softmax_row = nn.Softmax(dim=-2)

    def mask(self, dim: (int, int)) -> Tensor:
        a, b = dim
        mask = torch.ones(b, a)
        mask = torch.triu(mask, diagonal=0)
        mask = torch.log(mask.T)
        return mask.to(self.device)

    def forward(self, x_q, x_k, x_v):
        Q = torch.matmul(x_q.transpose(-1, -2).contiguous(), self.W_Q)
        K = torch.matmul(x_k.transpose(-1, -2).contiguous(), self.W_K)
        V = torch.matmul(x_v.transpose(-1, -2).contiguous(), self.W_V)
        if self.with_mask == True:
            Q += self.mask(Q.shape[-2:])
        A = torch.matmul(self.softmax_col(K.transpose(-1, -2).contiguous() / self.d_k), V)
        B = torch.matmul(self.softmax_row(Q / self.d_q), A)
        b, h, w, d = B.shape
        B = self.W_O(B.view(b, w, h * d))
        B = B.unsqueeze(dim=1).permute(0, 1, 3, 2)

        return B

    
class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()
    
    
class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
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
    def __init__(self, dim_C, dim_H, dim_W, dropout=0.3):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim_W)
        self.pw_conv1 = PointWiseConv(chan_in=dim_C)
        self.glu = GLU(dim=-2)
        self.dw_conv1d = DepthWiseConv1d(dim_C, dim_C*2, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(dim_C)
        self.swish = Swish()
        self.pw_conv2 = PointWiseConv(chan_in=dim_C)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs):
        x = inputs
        b, c, h, w = x.shape
        x = x.reshape(b, h, w*c)
        
        x = self.ln1(x)
        x = x.reshape(b, c, h, w)
        
        x = self.pw_conv1(x)
        x = self.glu(x)
        x = self.dw_conv1d(x)
        x = x.reshape(b, c, -1, w)
        
        x = self.bn(x)
        x = self.swish(x)
        x = self.pw_conv2(x)
        x = self.dropout(x)
        return x
    
    
class LAC(nn.Module):
    def __init__(self, dim_B=1, dim_C=1, dim_H=64, dim_W=256, device="cpu", dropout=0.1):
        super().__init__()
        self.lffn1 = LFFN(inputs_dim=(dim_B, dim_C, dim_H, dim_W), dim_hid=1024)
        self.do1 = nn.Dropout(dropout)
        self.mhlsa = MHLA2(num_heads=4, dim_input_q=dim_H, dim_input_kv=dim_H, dim_q=16, dim_k=16, device=device)
        self.do2 = nn.Dropout(dropout)
        self.conv_module = ConvModule(dim_C, dim_H, dim_W)
        self.do3 = nn.Dropout(dropout)
        self.lffn2 = LFFN(inputs_dim=(dim_B, dim_C, dim_H, dim_W), dim_hid=1024)
        self.do4 = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim_W)
        
    def forward(self, inputs):
        x = inputs
        x = x + 1/2 * self.do1(self.lffn1(x))
        x = x + self.do2(self.mhlsa(x, x, x))
        x = x + self.do3(self.conv_module(x))
        x = x + 1/2 * self.do4(self.lffn2(x))
        x = self.ln(x)
        return inputs + x
    
    
class Encoder(nn.Module):
    def __init__(self, n_encoders=2, device="cpu"):
        super().__init__()
        self.lacs = nn.Sequential(*[LAC(device=device) for i in range(n_encoders)])
        
    def forward(self, inputs):
        x = self.lacs(inputs)
        return x
    

class DecoderBlock(nn.Module):
    def __init__(self, dim_shape_tgt, dim_shape_mem, device="cpu", dropout=0.1):
        super().__init__()
        dim_B, dim_C, dim_H, dim_tgt = dim_shape_tgt
        dim_mem = dim_shape_mem[-1]
        dim_mem_2 = dim_shape_mem[-2]
        self.mhla_with_mask = MHLA2(num_heads=4, dim_input_q=dim_H, dim_input_kv=dim_H, dim_q=16, dim_k=16, mask=True, device=device)
        self.do1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(dim_tgt)
        self.mhla_with_memory = MHLA2(num_heads=4, dim_input_q=dim_H, dim_input_kv=dim_mem_2, dim_q=16, dim_k=16, device=device)
        self.do2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(dim_tgt)
        self.lffn = LFFN(inputs_dim=(dim_B, dim_C, dim_H, dim_mem), dim_hid=1024)
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
    def __init__(self, dropout=0.3, device="cpu", n_decoders=4):
        super().__init__()
        self.device = device
        self.lin1 = nn.Linear(152, 256).to(device)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(38, 64).to(device)
        self.relu2 = nn.ReLU()
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dim_shape_tgt=(1, 1, 64, 256), dim_shape_mem=(1, 1, 64, 256), device=device)
            for _ in range(n_decoders)])
        self.classifier = nn.Sequential(
            nn.Linear(64, 38).to(device),
            nn.Dropout(dropout),
        )
        
    def forward(self, mem, tgt):
        y = tgt.to(self.device)
        y = self.lin1(y)
        y = self.relu1(y)
        y = y.transpose(-1, -2)
        y = self.lin2(y)
        y = self.relu2(y)
        y = y.transpose(-1, -2)
        
        for dec in self.dec_blocks:
            y = dec(mem, y)
        
        y = y.transpose(-1, -2)
        y = self.classifier(y)
        return y

class InputPreprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_dim = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2),
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=2, stride=3),
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2)
        )
        self.conv_time = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2, stride=2)
        )

    def forward(self, inputs):
        b, c, h, w = inputs.shape
        X = inputs.view(b, c*h, w)
        X = X.transpose(-1, -2).contiguous()
        X = self.conv_dim(X)
        X = X.transpose(-1, -2).contiguous()
        X = self.conv_time(X)
        b, h, w = X.shape
        X = X.view(b, 1, h, w)
        return X

    
class Conformer(nn.Module):
    def __init__(self, n_encoders=2, n_decoders=2, device="cpu", dropout=0.3):
        super().__init__()
        self.input_preprocess = InputPreprocessor()
        self.pos_enc_inp = AbsolutePositionEncoding()
        self.pos_enc_out = AbsolutePositionEncoding()
        self.encoder = Encoder(n_encoders=n_encoders, device=device)
        self.enc_lin1 = nn.Sequential(
            nn.Linear(64, 38),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.enc_lin2 = nn.Sequential(
            nn.Linear(256, 152),
            nn.ReLU(),
            nn.LogSoftmax(dim=-1)
        )
        self.decoder = Decoder(n_decoders=n_decoders, device=device)
        self.device = device
        self.to(device)

    def to(self, device, *args, **kwargs):
        self = super().to(device, *args, **kwargs)
        self.device = device
        return self

    def forward(self, inputs, tgt):
        x = self.input_preprocess(inputs)
        x = x + self.pos_enc_inp(x).to(self.device)
        x = self.encoder(x)        
        y = self.pos_enc_out(tgt)
        y = self.decoder(x, y)
        x = x.transpose(-1, -2).contiguous()
        x = self.enc_lin1(x)
        x = x.transpose(-1, -2).contiguous()
        x = self.enc_lin2(x)

        return x, y


class CommandClassifier(Conformer):
    def __init__(self, *args, **kwargs):
        super(CommandClassifier, self).__init__(*args, **kwargs)
        self.lin_out = nn.Sequential(nn.Linear(38 * 256, 5),
                                  nn.Dropout(0.3),
                                  nn.Softmax(dim=-1)).to(kwargs["device"])

    def forward(self, inputs, tgt):
        x, y = super(CommandClassifier, self).forward(inputs, tgt)
        b, _, t, d = y.shape
        out = y.view(b, t * d)
        out = self.lin_out(out)
        return x, out


class CommandClassifierByEncoder(Conformer):
    def __init__(self, *args, **kwargs):
        super(CommandClassifierByEncoder, self).__init__(*args, **kwargs)
        self.lin_out = nn.Sequential(nn.Linear(38 * 152, 5),
                                  nn.Dropout(0.3),
                                  nn.Softmax(dim=-1)).to(kwargs["device"])

    def forward(self, inputs, tgt):
        x, y = super(CommandClassifierByEncoder, self).forward(inputs, tgt)
        b, _, t, d = x.shape
        out = x.view(b, t * d)
        out = self.lin_out(out)
        return out, out