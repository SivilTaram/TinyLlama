import os
import json
import random
from tqdm import tqdm
import re
from functools import partial
import multiprocessing

VOCAB_FILE = '../../id-en.txt'
print('Loading vocabulary from {}'.format(VOCAB_FILE))

def load_indonesian_english_dictionary(dict_file_path):
    """
    Load a dictionary of Indonesian-English word pairs.
    """
    with open(dict_file_path, 'r') as f:
        lines = f.readlines()
    indonesian_en_dict = {}
    counter = 0
    for line in lines:
        indonesian_word, english_word = line.strip().split("\t")
        # remove those words which are equal to themselves
        if indonesian_word != english_word:
            # if not in the dict
            if indonesian_word not in indonesian_en_dict:
                indonesian_en_dict[indonesian_word] = []
            indonesian_en_dict[indonesian_word].append(english_word)
            counter += 1
    print("Total {} words in the dictionary".format(counter))
    return indonesian_en_dict

id_en_dict = load_indonesian_english_dictionary(VOCAB_FILE)
replace_counter = 0
all_counter = 0
    
# for id_en_dict's keys, compile their regex in advance
word_prefix = set()
for key in id_en_dict.keys():
    word_prefix.add(key.split()[0])

def random_replace(sentence, replace_prob=0.2):
    """
    Randomly replace words in a sentence with a placeholder token.
    """
    words = sentence.split()
    global replace_counter
    global all_counter
    
    all_counter += len(words)
    
    new_words = []
    i = 0
    while i < len(words):
        if random.random() < replace_prob and words[i] in word_prefix:
            check_flag = False
            remaining_phrases = ' '.join(words[i:])  # 从当前位置到字符串末尾的子短语
            for check_phrase in id_en_dict.keys():
                if remaining_phrases.startswith(check_phrase):
                    # take one word from the list of possible replacements
                    replace_phrase = random.choice(id_en_dict[check_phrase])
                    new_words.append(replace_phrase)
                    i += len(check_phrase.split())
                    replace_counter += 1
                    check_flag = True
                    break
            if not check_flag:
                new_words.append(words[i])
                i += 1
        else:
            new_words.append(words[i])
            i += 1

    new_sentence = ' '.join(new_words)
    return new_sentence

def make_code_switch_process(file_name, data_folder_path, output_folder_path):
    file_path = os.path.join(data_folder_path, file_name)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Replace words in each line
        replace_lines = []
        for line in lines:
            example = json.loads(line)
            new_line = random_replace(example['text'], replace_prob=0.15)
            replace_lines.append(new_line)
            if len(replace_lines) % 1000 == 0:
                print("Replace {} words in {} words".format(replace_counter, all_counter))
        # Write to output file
        output_file_path = os.path.join(output_folder_path, file_name)
        with open(output_file_path, 'w') as f:
            for line in replace_lines:
                f.write(json.dumps({"text": line}, ensure_ascii=False) + '\n')


def build_code_switch_indonesian_corpus(data_folder_path, output_folder_path):
    """
    Build a corpus of code-switched Indonesian-English sentences.
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    all_files = os.listdir(data_folder_path)
    process_func = partial(make_code_switch_process, data_folder_path=data_folder_path, output_folder_path=output_folder_path)
    # use multi-processing to speed up
    with multiprocessing.Pool(64) as pool:
        pool.map(process_func, all_files)
        

if __name__ == '__main__':
    # build_code_switch_indonesian_corpus('/home/aiops/liuqian/TinyLlama/hf_dataset/redpajama_20b/train', 
    #                                     '/home/aiops/liuqian/TinyLlama/hf_dataset/redpajama_20b_sen_switch_0.5/train')
    # build_code_switch_indonesian_corpus('/home/aiops/liuqian/TinyLlama/hf_dataset/redpajama_20b/valid', 
    #                                 '/home/aiops/liuqian/TinyLlama/hf_dataset/redpajama_20b_sen_switch_0.5/valid')
    # build_code_switch_indonesian_corpus('/home/aiops/liuqian/TinyLlama/hf_dataset/cleaned_cc100_ind/train', 
    #                                     '/home/aiops/liuqian/TinyLlama/hf_dataset/cleaned_cc100_word_switch_0.2_new/train')
    # build_code_switch_indonesian_corpus('/home/aiops/liuqian/TinyLlama/hf_dataset/cleaned_cc100_ind/valid', 
    #                                 '/home/aiops/liuqian/TinyLlama/hf_dataset/cleaned_cc100_word_switch_0.2_new/valid')
    # build_code_switch_indonesian_corpus('/home/aiops/liuqian/TinyLlama/hf_dataset/cleaned_cc100_en_ind/train', 
    #                                     '/home/aiops/liuqian/TinyLlama/hf_dataset/cleaned_en_ind_word_switch_0.2/train')
    build_code_switch_indonesian_corpus('/home/aiops/liuqian/TinyLlama/hf_dataset/cleaned_cc100_ind/train', 
                                        '/home/aiops/liuqian/TinyLlama/hf_dataset/cleaned_cc100_word_switch_0.1_new/train')
