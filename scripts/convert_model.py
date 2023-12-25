import torch
import torch.nn as nn
import os
from transformers import LlamaTokenizer
# use sentencepiece to tokenize the text
import sentencepiece as spm
# 加载你的state_dict文件
state_dict_path = '../../TinyLlama-1T-Model/pytorch_model.bin'
old_tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
new_tokenizer = spm.SentencePieceProcessor(model_file='../data/new_llama/tokenizer.model')
state_dict = torch.load(state_dict_path)

print(state_dict.keys())
print(state_dict['model.embed_tokens.weight'].shape)
print(state_dict['lm_head.weight'].shape)

# print part of the embedding
print(state_dict['model.embed_tokens.weight'][0][:10])
print(state_dict['lm_head.weight'][0][:10])

for key in ['model.embed_tokens.weight', 'lm_head.weight']:
    # 获取原始的embedding权重
    original_embedding_weight = state_dict[key]
    original_embedding_dim = original_embedding_weight.shape[0]
    hidden_size = original_embedding_weight.shape[1]
    expanded_embedding_dim = 45120

    # 计算原始embedding的平均值
    stddev = 0.01  # 标准差可以根据实际情况调整
    mean_embedding = original_embedding_weight.mean(dim=0)
    # the shape should be [diff, 2048]
    gaussian_noise = torch.randn(expanded_embedding_dim - original_embedding_dim, hidden_size) * stddev    

    # new_embeddings = mean_embedding + gaussian_noise
    new_embeddings = torch.zeros(expanded_embedding_dim - original_embedding_dim, hidden_size)
    for i in range (32001, 45071):
        cur_embedding = new_embeddings[i - 32001]
        # get the token from the new tokenizer
        token = new_tokenizer.decode(i)
        # get the index from the old tokenizer
        indexes = old_tokenizer.encode(token)
        if indexes[1] == 29871:
            indexes = indexes[2:]
        else:
            indexes = indexes[1:]
        print(token, indexes)
        # get the average embedding from the old embedding
        for i in range(len(indexes)):
            cur_embedding += original_embedding_weight[indexes[i]]
        cur_embedding /= len(indexes)

    initialized_embedding = torch.cat([original_embedding_weight, new_embeddings], dim=0)
    # 更新state_dict中的embedding权重
    state_dict[key] = initialized_embedding

print(state_dict['model.embed_tokens.weight'].shape)
print(state_dict['lm_head.weight'].shape)

print(state_dict['model.embed_tokens.weight'][0][:100])
print(state_dict['lm_head.weight'][0][:100])

print(state_dict['model.embed_tokens.weight'][45000][:100])
print(state_dict['lm_head.weight'][45000][:100])

# save the new state_dict
if not os.path.exists('../../TinyLlama-1T-Model-Semantic'):
    os.makedirs('../../TinyLlama-1T-Model-Semantic')
torch.save(state_dict, '../../TinyLlama-1T-Model-Semantic/pytorch_model.bin')