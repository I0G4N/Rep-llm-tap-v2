from model.transformer import Transformer
from torch import nn
import torch
import time
from torch.nn import functional as F
import numpy as np
from utils.process import read_data, create_fields, create_dataset, tokenize_en
from utils.batch import create_masks


# data
src_file = 'data/english.txt'
trg_file = 'data/french.txt'
src_lang = 'en_core_web_trf'
trg_lang = 'fr_dep_news_trf'
max_strlen = 80 # max sequence length
batchsize = 1500
src_data, trg_data = read_data(src_file, trg_file)
EN_TEXT, FR_TEXT = create_fields(src_lang, trg_lang)
train_iter, src_pad, trg_pad = create_dataset(src_data, trg_data, EN_TEXT, FR_TEXT, max_strlen, batchsize).to('cuda' if torch.cuda.is_available() else 'cpu')


# model parameters
d_model = 512
heads = 8
N = 6
src_vocab = len(EN_TEXT.vocab)
trg_vocab = len(FR_TEXT.vocab)

model = Transformer(src_vocab, trg_vocab, d_model, N, heads).to('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the parameters of the model
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

# Define the optimizer
optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9).to('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(epochs, print_every=1000):

    model.train()

    start = time.time()
    temp = start

    total_loss = 0

    for epoch in range(epochs):
        for i, batch in enumerate(train_iter):
            # src: (max_seq_len, batch_size) -> (batch_size, max_seq_len)
            src = batch.src.transpose(0, 1)
            trg = batch.trg.transpose(0, 1)

            # Remove <eos> token
            # (batch_size, seq_len) -> (batch_size, trg_seq_len(max_seq_len - 1))
            trg_input = trg[:, :-1] # from <sos> to the last token before <eos> or <pad>

            # remove <sos> token from target
            # from the next token of <sos> to <eos>, then flatten
            # (batch_size * trg_seq_len)
            targets = trg[:, 1:].contiguous().view(-1)

            # (batch_size, 1, src_seq_len), (batch_size, 1, trg_seq_len)
            src_mask, trg_mask = create_masks(src, trg_input, src_pad, trg_pad)

            # predictions of the model, scores of the target vocabulary for each token in the target sequence based on the previous masked tokens
            # (batch_size, trg_seq_len, trg_vocab)
            preds = model(src, trg_input, src_mask, trg_mask)

            optim.zero_grad()

            # (batch_size, trg_seq_len, trg_vocab) -> (batch_size * trg_seq_len, trg_vocab)
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)),
                                   targets, ignore_index=trg_pad)
            
            loss.backward()
            optim.step()

            total_loss += loss.item()
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print("time = %dm, epoch = %d, iter = %d, loss = %.3f, %ds per %d iters" %
                      ((time.time() - start) // 60, epoch + 1, i + 1, loss_avg,
                       (time.time() - temp),
                       print_every))
                total_loss = 0
                temp = time.time()


def translate(src, max_len=80, custom_string=False):
    model.eval()
    if custom_string:
        src = tokenize_en(src, EN_TEXT)
        src = torch.LongTensor(src)
    print(src)
    src_mask = (src != src_pad).unsqueeze(-2)
    e_outputs = model.encoder(src.unsqueeze(0), src_mask)

    outputs = torch.zeros(max_len).type_as(src.data)
    outputs[0] = torch.LongTensor([FR_TEXT.vocab.stoi["<sos>"]])

    for i in range(1, max_len):
        trg_mask = np.triu(np.ones((1, i, i)).astype('uint8'))
        trg_mask = torch.from_numpy(trg_mask) == 0

        out = model.out(model.decoder(outputs[:i].unsqueeze(0),
                                      e_outputs, src_mask, trg_mask))
        
        out = F.softmax(out, dim=-1)
        val, ix = out[:, :-1].data.topk(1)

        outputs[i] = ix[0][0]
        if outputs[i] == FR_TEXT.vocab.stoi["<eos>"]:
            break

    return " ".join([FR_TEXT.vocab.itos[ix] for ix in outputs[:i]])