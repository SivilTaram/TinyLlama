import os
import json
import random
from tqdm import tqdm

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

def random_replace(sentence, replace_prob=0.2):
    """
    Randomly replace words in a sentence with a placeholder token.
    """
    words = sentence.split()
    for i in range(len(words)):
        if random.random() < replace_prob and words[i].lower() in id_en_dict:
            # random take a word from the dict
            words[i] = random.choice(id_en_dict[words[i].lower()])
    new_sentence = ' '.join(words)
    return new_sentence

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
                new_line = random_replace(example['text'], replace_prob=0.5)
                replace_lines.append(new_line)
            # Write to output file
            output_file_path = os.path.join(output_folder_path, file_name)
            with open(output_file_path, 'w') as f:
                for line in replace_lines:
                    f.write(json.dumps({"text": line}, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    # build_code_switch_indonesian_corpus('/home/aiops/liuqian/TinyLlama/hf_dataset/redpajama_20b/train', 
    #                                     '/home/aiops/liuqian/TinyLlama/hf_dataset/redpajama_20b_sen_switch_0.5/train')
    # build_code_switch_indonesian_corpus('/home/aiops/liuqian/TinyLlama/hf_dataset/redpajama_20b/valid', 
    #                                 '/home/aiops/liuqian/TinyLlama/hf_dataset/redpajama_20b_sen_switch_0.5/valid')
    # build_code_switch_indonesian_corpus('/home/aiops/liuqian/TinyLlama/hf_dataset/cleaned_cc100_ind/train', 
    #                                     '/home/aiops/liuqian/TinyLlama/hf_dataset/cleaned_cc100_word_switch_0.2_new/train')
    # build_code_switch_indonesian_corpus('/home/aiops/liuqian/TinyLlama/hf_dataset/cleaned_cc100_ind/valid', 
    #                                 '/home/aiops/liuqian/TinyLlama/hf_dataset/cleaned_cc100_word_switch_0.2_new/valid')
    build_code_switch_indonesian_corpus('/home/aiops/liuqian/TinyLlama/hf_dataset/cleaned_cc100_ind/train', 
                                        '/home/aiops/liuqian/TinyLlama/hf_dataset/cleaned_cc100_word_switch_0.5_new/train')
    build_code_switch_indonesian_corpus('/home/aiops/liuqian/TinyLlama/hf_dataset/cleaned_cc100_ind/valid', 
                                    '/home/aiops/liuqian/TinyLlama/hf_dataset/cleaned_cc100_word_switch_0.5_new/valid')