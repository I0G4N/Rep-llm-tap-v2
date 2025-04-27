import torch
from torch import nn


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_seq_len, d_model) # (max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = torch.sin(pos / (10000 ** (i / d_model)))
                pe[pos, i + 1] = torch.cos(pos / (10000 ** (i / d_model)))

        pe = pe.unsqueeze(0) # (1, max_seq_len, d_model) add batch dimension
        # not a parameter, but a buffer which means it won't be updated during training
        self.register_buffer('pe', pe) # (1, max_seq_len, d_model)

    def forward(self, x):
        # scale the value of input 
        x = x * torch.sqrt(self.d_model) # (batch_size, seq_len, d_model)
        seq_len = x.size(1)  
        # add positional encoding to input
        x = x + self.pe[:, :seq_len].detach().cuda() # (batch_size, seq_len, d_model)
        return x
