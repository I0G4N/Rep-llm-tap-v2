import spacy
import re


class tokenize(object):
    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def tokenizer(self, sentence):
        sentence = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence)) # remove special characters
        sentence = re.sub(r"[ ]+", " ", sentence) # remove multiple spaces
        sentence = re.sub(r"\!+", "!", sentence) # remove multiple exclamation marks
        sentence = re.sub(r"\,+", ",", sentence) # remove multiple commas
        sentence = re.sub(r"\?+", "?", sentence) # remove multiple question marks
        sentence = sentence.lower() # convert to lowercase
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]
    