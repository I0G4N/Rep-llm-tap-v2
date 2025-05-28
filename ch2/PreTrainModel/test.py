from transformers import BertForMaskedLM, BertTokenizerFast, pipeline
import os

model = BertForMaskedLM.from_pretrained(os.path.join("pretrained-bert", "checkpoint-10000"))
tokenizer = BertTokenizerFast.from_pretrained("pretrained-bert")

# The task type is designated as "fill-in-the-blank", which requires the model to predict the most likely word at the masked position in the text
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

examples = [
    "The capital of France is [MASK].",
    "The capital of Germany is [MASK].",
    "The capital of Italy is [MASK].",
    "The capital of Spain is [MASK].",
    "The capital of Portugal is [MASK]."
]

for example in examples:
    for prediction in fill_mask(example):
        print(f"Input: {example}")
        print(f"Prediction: {prediction['token_str']}, Score: {prediction['score']:.4f}")
        print("-" * 50)