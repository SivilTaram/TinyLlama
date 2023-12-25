import sentencepiece as spm

# Load the SentencePiece tokenizer model
model_path = '../data/mistral_sea/tokenizer.model'
sp = spm.SentencePieceProcessor()
sp.load(model_path)

# old_model_path = '../data/llama/tokenizer.model'
# old_sp = spm.SentencePieceProcessor()
# old_sp.load(old_model_path)

vocab_size = sp.get_piece_size()
print("Vocabulary size:", vocab_size)
# for i in range(vocab_size):
#     if i < 40000:
#         continue
#     token_str = sp.decode(i)
#     print(token_str)
#     # use old tokenizer to get the index
#     index = old_sp.encode(token_str)
#     print(index)
    
