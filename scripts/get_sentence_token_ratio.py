import sentencepiece as spm
import os
from tqdm import tqdm
import json

# Load the SentencePiece tokenizer model
model_path = '../data/mistral_sea/tokenizer.model'
sp = spm.SentencePieceProcessor()
sp.load(model_path)

def tokenize_sentence(sentence):
    return sp.encode(sentence)

def count_tokens(sentence):
    if isinstance(sentence, str):
        sentence = tokenize_sentence(sentence)
        return len(sentence)
    else:
        token_list = tokenize_sentence(sentence)
        return sum([len(token) for token in token_list])

def count_file_tokens(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [json.loads(line)['text'] for line in lines]
        total_tokens = 0
        for idx in tqdm(range(0, len(lines), 10000)):
            total_tokens += count_tokens(lines[idx:idx+10000])
        print("Total tokens:", total_tokens)
        print("Total sentences:", len(lines))
        print("Tokens per sentence:", total_tokens / len(lines))

# file_paths = os.listdir('/home/aiops/liuqian/TinyLlama/hf_dataset/data_tokenize')
# # sorted by name
# file_paths.sort()
# for file_path in file_paths:
#     print(file_path)
#     count_file_tokens('/home/aiops/liuqian/TinyLlama/hf_dataset/data_tokenize/' + file_path)
#     print("====================================")
count_file_tokens('/home/aiops/liuqian/TinyLlama/hf_dataset/data_mixture/train/en_redpajama_1.jsonl')