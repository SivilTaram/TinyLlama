import os
import json
import random
from tqdm import tqdm

def load_indonesian_english_dictionary(dict_file_path):
    """
    Load a dictionary of Indonesian-English word pairs.
    """
    with open(dict_file_path, 'r') as f:
        lines = f.readlines()
    indonesian_en_dict = {}
    for line in lines:
        indonesian_word, english_word = line.strip().split("\t")
        indonesian_en_dict[indonesian_word] = english_word
    return indonesian_en_dict

id_en_dict = load_indonesian_english_dictionary('../../en-id.txt')

def random_replace(sentence, replace_prob=0.2):
    """
    Randomly replace words in a sentence with a placeholder token.
    """
    words = sentence.split()
    for i in range(len(words)):
        if random.random() < replace_prob and words[i] in id_en_dict:
            words[i] = id_en_dict[words[i]]
    return ' '.join(words)

def build_code_switch_indonesian_corpus(data_folder_path, output_folder_path):
    """
    Build a corpus of code-switched Indonesian-English sentences.
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    all_files = os.listdir(data_folder_path)
    for file_name in tqdm(all_files):
        # Read the file
        file_path = os.path.join(data_folder_path, file_name)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # Replace words in each line
            replace_lines = []
            for line in lines:
                example = json.loads(line)
                if random.random() < 0.2:
                    line = random_replace(example['text'], replace_prob=0.2)
                else:
                    line = example['text'].strip()
                replace_lines.append(line)
            # Write to output file
            output_file_path = os.path.join(output_folder_path, file_name)
            with open(output_file_path, 'w') as f:
                for line in replace_lines:
                    f.write(json.dumps({"text": line}, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    build_code_switch_indonesian_corpus('/home/aiops/liuqian/TinyLlama/hf_dataset/redpajama_20b/train', 
                                        '/home/aiops/liuqian/TinyLlama/hf_dataset/redpajama_20b_sen_switch_0.2/train')
    build_code_switch_indonesian_corpus('/home/aiops/liuqian/TinyLlama/hf_dataset/redpajama_20b/valid', 
                                    '/home/aiops/liuqian/TinyLlama/hf_dataset/redpajama_20b_sen_switch_0.2/valid')