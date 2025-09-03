from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
sen = "你好呀"
inputs = tokenizer(sen, padding="max_length", max_length=15)
print(inputs)

# {
# "input_ids": [101, 2483, 2207, 4638, 2769, 738, 3300, 1920, 3457, 2682, 106, 102, 0, 0, 0], 
# "token_type_ids": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
# "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
# }
# inputs_ids: embeddings,
# token_type_ids: segment_ids,
# attention_mask: padding part.