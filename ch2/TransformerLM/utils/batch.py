import torch
from torchtext import data
import numpy as np
from torch.autograd import Variable


def nopeak_mask(size):
    """Create a mask to prevent attention to future tokens.
    Args:
        size: size of the mask (sequence length)
    Returns:
        A boolean mask of shape (1, 1, size, size) where future positions are set to True.
    """
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8') # set as int
    np_mask = Variable(torch.from_numpy(np_mask) == 0) # set to boolean matrix

    return np_mask


def create_masks(src, trg, src_pad, trg_pad):
    """Create masks for source and target sequences.
    Args:
        src: source sequence tensor (batch_size, src_seq_len)
        trg: target sequence tensor (batch_size, trg_seq_len)
        src_pad: padding index for source
        trg_pad: padding index for target
    """
    src_mask = (src != src_pad).unsqueeze(-2) # (batch_size, 1, src_seq_len)

    if trg is not None:
        trg_mask = (trg != trg_pad).unsqueeze(-2)
        size = trg.size(1)
        np_mask = nopeak_mask(size)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None
    return src_mask, trg_mask # (batch_size, 1, src_seq_len), (batch_size, 1, trg_seq_len)


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                """
                Arguments:
                    d: list of data
                    random_shuffler: function to shuffle data
                """
                for p in data.batch(d, self.batch_size * 100): # split data into big batches
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key), # sort the big batch by sort_key
                        self.batch_size, self.batch_size_fn # split into smaller batches and apply batch_size_fn
                    )
                    for b in random_shuffler(list(p_batch)): # randomly shuffle the smaller batches
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    """Keep augmenting batch and calculate total number of tokens + padding.
    Args:
        new: new example to be added to the batch
        count: number of examples in the current batch
        sofar: total number of tokens in the current batch"""
    global max_src_in_batch, max_tgt_in_batch
    if count == 0: # when the first example is added to the batch, reset the max values
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 2) # include <sos> and <eos> tokens in target length
    src_elements = count * max_src_in_batch # total number of source tokens in the batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements) # return the sum of the maximum of source and target tokens in the batch