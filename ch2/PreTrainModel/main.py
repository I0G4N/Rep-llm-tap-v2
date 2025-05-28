from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os, json
from datasets import concatenate_datasets, load_dataset
from itertools import chain


bookcorpus = load_dataset("bookcorpus", split="train")
wiki = load_dataset("wikipedia", "20220301.en", split="train")

wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])

dataset = concatenate_datasets([bookcorpus, wiki])

d = dataset.train_test_split(test_size=0.1)


def dataset_to_text(dataset, output_filename="data.txt"):
    with open(output_filename, "w") as f:
        for t in dataset["text"]:
            print(t, file=f)


dataset_to_text(d["train"], "train.txt")
dataset_to_text(d["test"], "test.txt")

special_tokens_dict = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
] # padding, unknown, classification, separation, mask, start, end

files = ["train.txt"]
vocab_size = 30_522 # vocab size of BERT
max_length = 512 # max length of BERT
truncate = False # truncate the text if it is longer than max_length

tokenizer = BertWordPieceTokenizer()
tokenizer.train(
    files,
    vocab_size=vocab_size,
    special_tokens=special_tokens_dict
)
tokenizer.enable_truncation(max_length=max_length) # truncate the text when longer than max_length

model_path = "pretrained-bert"

if not os.path.isdir(model_path):
    os.makedirs(model_path)
tokenizer.save_model(model_path)

with open(os.path.join(model_path, "config.json"), "w") as f:
    tokenizer_config = {
        "do_lower_case": True,
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
        "model_max_length": max_length,
    }
    json.dump(tokenizer_config, f)

# Load the tokenizer with transformers
tokenizer = BertTokenizerFast.from_pretrained(model_path)


def encode_with_truncation(examples):
    return tokenizer(examples["text"],
                      truncation=True,
                      max_length=max_length,
                      padding="max_length",
                      return_special_tokens_mask=True)


def encode_without_truncation(examples):
    return tokenizer(examples["text"],
                      truncation=False,
                      return_special_tokens_mask=True)


encode = encode_with_truncation if truncate else encode_without_truncation

train_dataset = d["train"].map(encode, batched=True)
test_dataset = d["test"].map(encode, batched=True)

if truncate:
    # only input_ids and attention_mask are needed for training, set format to torch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
else:
    # remove other columns but keep format
    train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
    test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])


def group_texts(examples):
    # Concatenate all texts, flatten then split into chunks of max_length
    # flatten the original nested lists into a single list
    concatenated_examples = {k: list(chain(examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]]) # length of token
    # drop the small remainder
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    # Split by chunks of max_length
    result = {
        k: [v[i : i + max_length] for i in range(0, total_length, max_length)]
        for k, v in concatenated_examples.items()
    }
    return result

# batched=True is used to process multiple examples at once
# tokenizer.enable_truncation(max_length=max_length)

if not truncate:
    # group the texts into chunks of max_length
    train_dataset = train_dataset.map(group_texts, batched=True,
                                      desc=f"Grouping texts into chunks of {max_length}")
    test_dataset = test_dataset.map(group_texts, batched=True,
                                    desc=f"Grouping texts into chunks of {max_length}")
    # set the format to torch
    train_dataset.set_format(type="torch")
    test_dataset.set_format(type="torch")


model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
model = BertForMaskedLM(config=model_config) # create a BERT model for masked language modeling

# Initialize the data collator for language modeling, randomly masks tokens in the input by the prob
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.2
)

training_args = TrainingArguments(
    output_dir=model_path,  # output directory to save the model checkpoints
    eval_strategy="steps",  # evaluate the model every logging_steps
    overwrite_output_dir=True,  # overwrite the output directory if it exists
    num_train_epochs=10,
    per_device_train_batch_size=10,
    gradient_accumulation_steps=8, # accumulate gradients over 8 steps

    per_device_eval_batch_size=64,
    logging_steps=1000,  # log every 1000 steps
    save_steps=1000,  # save the model every 1000 steps
    load_best_model_at_end=True,  # load the best model at the end of training
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()  # start training