import pandas as pd
from torchtext import data
from utils.tokenizer import tokenize
from utils.batch import MyIterator, batch_size_fn


def read_data(src_file, trg_file):
    if src_file is not None:
        try:
            src_data = open(src_file).read().strip().split('\n')
            # read all lines, remove leading/trailing whitespace, and split by newline
        except:
            print("error: '" + src_file + "' file not found.")
            quit()

    if trg_file is not None:
        try:
            trg_data = open(trg_file).read().strip().split('\n')
        except:
            print("error: '" + trg_file + "' file not found.")
            quit()
    
    return src_data, trg_data


def create_fields(src_lang, trg_lang):
    spacy_langs = ['en_core_web_trf', 'fr_dep_news_trf']
    if src_lang not in spacy_langs:
        print("invalid src language: " + src_lang + 'supported languages: ' + str(spacy_langs))
    if trg_lang not in spacy_langs:
        print("invalid trg language: " + trg_lang + 'supported languages: ' + str(spacy_langs))

    print("loading spacy tokenizers...")

    t_src = tokenize(src_lang)
    t_trg = tokenize(trg_lang)

    # Create fields (processing pipelines) for source and target languages
    TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=t_src.tokenizer)

    return (SRC, TRG)


def create_dataset(src_data, trg_data, SRC, TRG, max_strlen, batchsize):
    print("creating dataset and iterator...")

    raw_data = {'src': [line for line in src_data], 'trg': [line for line in trg_data]}
    df = pd.DataFrame(raw_data, columns=['src', 'trg'])

    # count of spaces in each sentence is equal to the number of words minus one
    # so we can use it to filter sentences by length
    mask = (df['src'].str.count(' ') < max_strlen) & (df['trg'].str.count(' ') < max_strlen)
    df = df.loc[mask]

    df.to_csv('translate_transformer_temp.csv', index=False)

    data_fields = [('src', SRC), ('trg', TRG)]
    # get padding indices for source and target languages
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

    train_iter = MyIterator(train, batch_size=batchsize, device='cuda',
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True, shuffle=True)
    
    # build vocabulary table for source and target languages
    SRC.build_vocab(train)
    TRG.build_vocab(train)

    # get the padding indices in the vocabulary of source and target languages
    src_pad = SRC.vocab.stoi['<pad>']
    trg_pad = TRG.vocab.stoi['<pad>']

    return train_iter, src_pad, trg_pad


def tokenize_en(src, SRC):
    """
    Tokenizes English text from a string into a list of tokens.
    """
    spacy_en = tokenize('en_core_web_trf') # tokenizer
    src = spacy_en.tokenizer(src) # tokenize the input string
    return [SRC.vocab.stoi[tok] for tok in src] # convert tokens to indices
