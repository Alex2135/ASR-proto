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
    

class LightAttention(nn.Module):
    def __init__(self, dim_input_q, dim_input_kv, dim_q, dim_k, with_mask=False):
        super().__init__()
        self.with_mask = with_mask
        self.softmax_col = nn.Softmax(dim=-1)
        self.softmax_row = nn.Softmax(dim=-2)
        self.W_q = nn.Linear(in_features=dim_input_q, out_features=dim_q)
        self.W_k = nn.Linear(in_features=dim_input_kv, out_features=dim_k)
        self.W_v = nn.Linear(in_features=dim_input_kv, out_features=dim_k)
        self.d_q = torch.pow(torch.Tensor([dim_q]), 1/4)
        self.d_k = torch.pow(torch.Tensor([dim_k]), 1/4)
        
    def mask(self, dim: (int, int)) -> Tensor :
        a, b = dim
        mask = torch.ones(b, a)
        mask = torch.triu(mask, diagonal=0)
        mask = torch.log(mask.T)
        return mask
        
    def forward(self, x_q, x_k, x_v):
        Q = self.W_q(x_q)
        K = self.W_k(x_k)
        V = self.W_v(x_v)
        if self.with_mask == True:
            Q += self.mask(Q.shape[-2:])
        A = self.softmax_row(Q / self.d_q)
        B = torch.matmul(self.softmax_col(K.transpose(-2, -1) / self.d_k), V)
        Z = torch.matmul(A, B)
        
        return Z


class MHLA(nn.Module):
    def __init__(self, 
                 num_heads, 
                 dim_input_q,
                 dim_input_kv,
                 dim_q = 64,
                 dim_k = 64,
                 mask=False
                ):
        """
        Args:
        dim_input - if shape is (B, C, H, W), then dim_input is W
        """
        super().__init__()
        heads = [LightAttention(dim_input_q, dim_input_kv, dim_q, dim_k, mask) for _ in range(num_heads)]
        self.heads = nn.ModuleList(heads)                
        self.W_o = nn.Linear(dim_k*num_heads, dim_input_kv)
        
    def forward(self, x_q, x_k, x_v):
        x = torch.cat([latt(x_q, x_k, x_v) for latt in self.heads], dim=-1)
        y = self.W_o(x)
        return y
    
    
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
        self.ln1 = nn.LayerNorm([dim_H, dim_W*dim_C])
        self.pw_conv1 = PointWiseConv(chan_in=dim_C)
        self.glu = GLU(-2)
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
    def __init__(self, dim_B=1, dim_C=1, dim_H=64, dim_W=256):
        super().__init__()
        self.lffn1 = LFFN(inputs_dim=(dim_B, dim_C, dim_H, dim_W), dim_hid=1024)
        self.mhlsa = MHLA(num_heads=4, dim_input_q=dim_W, dim_input_kv=dim_W)    
        self.conv_module = ConvModule(dim_C, dim_H, dim_W)
        self.lffn2 = LFFN(inputs_dim=(dim_B, dim_C, dim_H, dim_W), dim_hid=1024)
        self.ln = nn.LayerNorm([dim_C, dim_H, dim_W])
        
    def forward(self, inputs):
        x = inputs
        x = x + 1/2 * self.lffn1(x)
        x = x + self.mhlsa(x, x, x)
        x = x + self.conv_module(x)
        x = x + 1/2 * self.lffn2(x)
        x = self.ln(x)
        return inputs + x
    
    
class Encoder(nn.Module):
    def __init__(self, lacs_n=2):
        super().__init__()
        self.lacs = nn.Sequential(*[LAC() for i in range(lacs_n)])
        
    def forward(self, inputs):
        x = self.lacs(inputs)
        return x
    

class DecoderBlock(nn.Module):
    def __init__(self, dim_shape_tgt, dim_shape_mem):
        super().__init__()
        dim_B, dim_C, dim_H, dim_tgt = dim_shape_tgt
        dim_mem = dim_shape_mem[-1]
        self.mhla_with_mask = MHLA(num_heads=2, dim_input_q=dim_tgt, dim_input_kv=dim_tgt, mask=True)
        self.ln1 = nn.LayerNorm([dim_C, dim_H, dim_tgt])
        self.mhla_with_memory = MHLA(num_heads=2, dim_input_q=dim_tgt, dim_input_kv=dim_mem)
        self.ln2 = nn.LayerNorm([dim_C, dim_H, dim_mem])
        self.lffn = LFFN(inputs_dim=(dim_B, dim_C, dim_H, dim_mem), dim_hid=1024)
        self.ln3 = nn.LayerNorm([dim_C, dim_H, dim_mem])
        
    def forward(self, mem, y):
        y = y + self.mhla_with_mask(y, y, y)
        y = self.ln1(y)
        y = y + self.mhla_with_memory(y, mem, mem)
        y = self.ln2(y)
        y = y + self.lffn(y)
        y = self.ln3(y)
        return y
    
    
class Decoder(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.lin1 = nn.Linear(152, 256)
        self.swish1 = Swish()
        self.lin2 = nn.Linear(38, 64)
        self.swish2 = Swish()
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dim_shape_tgt=(1, 1, 64, 256), dim_shape_mem=(1, 1, 64, 256)) 
            for _ in range(4)])
        self.classifier = nn.Sequential(
            nn.Linear(64, 38),
            nn.Dropout(dropout),
        )
        
    def forward(self, mem, tgt):
        y = tgt
        y = self.lin1(y)
        y = self.swish1(y)
        y = y.transpose(-1, -2)
        y = self.lin2(y)
        y = self.swish2(y)
        y = y.transpose(-1, -2)
        
        for dec in self.dec_blocks:
            y = dec(mem, y)
        
        y = y.transpose(-1, -2)
        y = self.classifier(y)
        #print("y shape:", y.shape)
        return y
    
    
class Conformer(nn.Module):
    def __init__(self, lacs_n=2):
        super().__init__()
        self.input_preprocess = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2),
            Swish(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1),
            Swish()
        )
        self.pos_enc_inp = AbsolutePositionEncoding()
        self.pos_enc_out = AbsolutePositionEncoding()
        self.encoder = Encoder(lacs_n)
        self.decoder = Decoder()
        
    def forward(self, inputs, tgt):
        x = self.input_preprocess(inputs)
        x = x + self.pos_enc_inp(x)
        x = self.encoder(x)        
        y = self.pos_enc_out(tgt)
        y = self.decoder(x, y)
        
        return y