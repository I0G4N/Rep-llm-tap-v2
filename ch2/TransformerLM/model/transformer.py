import torch
from torch import nn
from torch.nn import functional as F
from utils.get_clones import get_clones
from utils.embedder import Embedder
from torch.autograd import Variable


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model) # (max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = torch.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:  # to avoid index out of range
                    pe[pos, i + 1] = torch.cos(pos / (10000 ** (i / d_model)))

        pe = pe.unsqueeze(0) # (1, max_seq_len, d_model) add batch dimension
        # not a parameter, but a buffer which means it won't be updated during training
        self.register_buffer('pe', pe) # (1, max_seq_len, d_model)

    def forward(self, x):
        # scale the value of input 
        x = x * torch.sqrt(self.d_model) # (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # add positional encoding to input
        x = x + Variable(self.pe[:, :seq_len, :], require_grad=False) # (batch_size, seq_len, d_model)
        return self.dropout(x) # (batch_size, seq_len, d_model)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        # input dimension must be divisible by number of heads
        self.d_k = d_model // heads
        self.h = heads

        # linear layers for query, key and value, Wq, Wk, Wv
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(q, k, v, d_k, mask=None, dropout=None):
        """
        Args:
            q: query tensor of shape (batch_size, h, seq_len, d_k)
            k: key tensor of shape (batch_size, h, seq_len, d_k)
            v: value tensor of shape (batch_size, h, seq_len, d_k)
            d_k: dimension of key
            mask: mask tensor of shape (batch_size, 1, seq_len)
            
        Returns:
            output: attention output tensor of shape (batch_size, h, seq_len, d_k)"""
        # attention scores, q * k^T, scaled by sqrt(d_k) to prevent large values
        # scores: (batch_size, h, seq_len, d_k) * (batch_size, h, d_k, seq_len) -> (batch_size, h, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(d_k)

        # apply mask to scores to prevent attention to certain positions of blank tokens
        # then after applying softmax, the masked positions will have 0 attention
        # usually, the mask in encoder is used to prevent attention to padding tokens
        # the mask in decoder is used to prevent attention to future tokens
        if mask is not None:
            mask = mask.unsqueeze(1) # (batch_size, 1, seq_len) -> (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)

        # apply softmax to scores to get attention weights of each query from each key
        scores = F.softmax(scores, dim=-1)

        # apply dropout to attention weights
        if dropout is not None:
            scores = dropout(scores)

        # apply attention weights to value tensor
        # output: (batch_size, h, seq_len, seq_len) * (batch_size, h, seq_len, d_k) -> (batch_size, h, seq_len, d_k)
        output = torch.matmul(scores, v)

        return output
    
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: query tensor of shape (batch_size, seq_len, d_model)
            k: key tensor of shape (batch_size, seq_len, d_model)
            v: value tensor of shape (batch_size, seq_len, d_model)
            mask: mask tensor of shape (batch_size, 1, seq_len)
        Returns:
            output: attention output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size = q.size(0)

        # perform linear operation and split into h heads
        # q, k, v: (batch_size, seq_len, d_model) -> (batch_size, h, seq_len, d_k)
        q = self.q_linear(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # apply attention on all h heads
        # scores: (batch_size, h, seq_len, d_k)
        scores = self.attention(q, k, v, self.d_k, mask=mask, dropout=self.dropout)

        # concatenate heads and put through final linear layer
        # scores: (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, d_model)
        # why contiguous? after transpose, the tensor is not contiguous in memory
        # so we need to make it contiguous before reshaping
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.out(concat) # (batch_size, seq_len, d_model)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.linear2(x)

        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        # the epsilon value to prevent division by zero
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # (batch_size, seq_len, 1)
        std = x.std(-1, keepdim=True)

        norm = (self.alpha * (x - mean) / (std + self.eps)) + self.bias

        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, d_ff=2048):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attention = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask=mask)
        attn_output = self.dropout_1(attn_output)
        x = x + attn_output
        x = self.norm_1(x)
        ff_output = self.feed_forward(x)
        ff_output = self.dropout_2(ff_output)
        x = x + ff_output
        x = self.norm_2(x)
        return x
    

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout=0.1, d_ff=2048):
        super().__init__()
        self.N = N # number of layers
        self.embdding = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout=dropout, d_ff=d_ff), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask=None):
        x = self.embdding(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask=mask)
        return self.norm(x) # (batch_size, seq_len, d_model)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, d_ff=2048):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        self.attention_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attention_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
            enc_output: encoder output tensor of shape (batch_size, seq_len, d_model)
            src_mask: source mask tensor of shape (batch_size, 1, seq_len) for padding tokens
            tgt_mask: target mask tensor of shape (batch_size, 1, seq_len) for future tokens
        """
        attn_output_1 = self.attention_1(x, x, x, mask=tgt_mask)
        attn_output_1 = self.dropout_1(attn_output_1)
        x = x + attn_output_1
        x = self.norm_1(x)
        attn_output_2 = self.attention_2(x, enc_output, enc_output, mask=src_mask)
        attn_output_2 = self.dropout_2(attn_output_2)
        x = x + attn_output_2
        x = self.norm_2(x)
        ff_output = self.feed_forward(x)
        ff_output = self.dropout_3(ff_output)
        x = x + ff_output
        x = self.norm_3(x)

        return x
    

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout=0.1, d_ff=2048):
        super().__init__()
        self.N = N # number of layers
        self.embedding = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout=dropout, d_ff=d_ff), N)
        self.norm = Norm(d_model)

    def forward(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(tgt)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, enc_output, src_mask=src_mask, tgt_mask=tgt_mask)
        x = self.norm(x) # (batch_size, seq_len, d_model)
        
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout=0.1, d_ff=2048):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout, d_ff)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout, d_ff)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(trg, enc_output, src_mask, trg_mask)
        output = self.out(dec_output)

        return output # (batch_size, seq_len, trg_vocab)