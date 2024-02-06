import sentencepiece as spm
from transformers import AutoTokenizer
from tqdm import tqdm

model_path = "../data/qwen2"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# try to tokenize a file and report the tokens

with open("../website_demo.jsonl", "r", encoding="utf8") as f:
    lines = f.readlines()
    total_length = 0
    for line in tqdm(lines):
        result = tokenizer(line, return_length=True)
        total_length += result["length"][0]
    print(f"Total tokens: {total_length}")
# Load the SentencePiece tokenizer model
# model_path = '../data/qwen/tokenizer.model'
# sp = spm.SentencePieceProcessor()
# sp.load(model_path)

# old_model_path = '../data/llama/tokenizer.model'
# old_sp = spm.SentencePieceProcessor()
# old_sp.load(old_model_path)

# vocab_size = sp.get_piece_size()
# print("Vocabulary size:", vocab_size)
# for i in range(vocab_size):
#     if i < 40000:
#         continue
#     token_str = sp.decode(i)
#     print(token_str)
#     # use old tokenizer to get the index
#     index = old_sp.encode(token_str)
#     print(index)
    
