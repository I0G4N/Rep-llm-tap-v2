from transformers import AutoTokenizer
from collections import defaultdict


corpus = [
    "This is the HuggingFace Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Initialize a defaultdict to count word fractions
word_fraqs = defaultdict(int) # defaultdict(int) means that if a key is not found, it will return 0 by default

for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    # [("This", (0, 4)), ("Ġis", (5, 8)), ("Ġthe", (9, 13)), ("ĠHuggingFace", (14, 27)), ("ĠCourse", (28, 36)), (".", (36, 37))]
    # Ġ indicates a space before the word in the GPT-2 tokenizer 

    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_fraqs[word] += 1

# Create a list of unique letters in the corpus
alphabet = []

for word in word_fraqs.keys():
    for letter in word:
        if letter not in alphabet:
            alphabet.append(letter)
alphabet.sort()

# Add the end-of-text token to the vocabulary
vocab = ["<|endoftext|>"] + alphabet.copy()

# split the words into letters
splits = {word: [letter for letter in word] for word in word_fraqs.keys()}


def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_fraqs.items():
        split = splits[word]
        if len(split) == 1: # if the word is a single letter, we skip it
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


def merge_pair(a, b, splits):
    for word in word_fraqs:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1: # iterate through the split letters
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2:]
                # merge the pair
            else:
                i += 1
        splits[word] = split
    return splits


vocab_size = 50

merges = {}

# util vocabulary size reached
while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items(): # get the most frequent pair
        if max_freq is None or freq > max_freq:
            best_pair = pair
            max_freq = freq
    splits = merge_pair(*best_pair, splits=splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])

# Print the final vocabulary and merges
print("Final Vocabulary:")
for word in vocab:
    print(word)
print("\nMerges:")
for pair, merged in merges.items():
    print(f"{pair[0]} {pair[1]} -> {merged}")


def bpe_tokenize(text):
    pre_tokenize_result = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenize_result = [word for word, _ in pre_tokenize_result]
    splits = [[letter for letter in word] for word in pre_tokenize_result] # [[letters] of words]
    for pair, merge in merges.items(): # replace pairs in the splits
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2:]
                else:
                    i += 1
            splits[idx] = split # update the split in the list
    return sum(splits, []) # join the letters back into words

print(bpe_tokenize("This is not a token."))  # Example usage of the BPE tokenizer

