import os
from tqdm import tqdm
import random

# downsamples the indian dataset to match the english dataset
def merge_datasets(ind_folder_path, en_folder_path, new_folder_path):
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    # get the list of files
    ind_files = os.listdir(ind_folder_path)
    en_files = os.listdir(en_folder_path)
    # select 30% of the indonesia files
    random.shuffle(ind_files)
    ind_files = ind_files[:int(len(ind_files)*0.3)]
    # read all the files
    all_lines = []
    for file_path in tqdm(ind_files):
        with open(os.path.join(ind_folder_path, file_path), "r") as f:
            all_lines.extend(f.readlines())
    # read all the files
    for file_path in tqdm(en_files):
        with open(os.path.join(en_folder_path, file_path), "r") as f:
            all_lines.extend(f.readlines())
    # shuffle the lines
    random.shuffle(all_lines)
    # write to the new folder
    chunk_idx = 0
    for i in tqdm(range(0, len(all_lines), 1000000)):
        chunk = all_lines[i:i+1000000]
        with open(f"{new_folder_path}/chunk_{chunk_idx}.jsonl", "w", encoding="utf8") as f:
            f.writelines(chunk)
        chunk_idx += 1

if __name__ == "__main__":
    merge_datasets("/home/aiops/liuqian/TinyLlama/hf_dataset/cleaned_cc100_word_switch_0.1_new/train",
                   "/home/aiops/liuqian/TinyLlama/hf_dataset/redpajama_20b/train",
                   "/home/aiops/liuqian/TinyLlama/hf_dataset/cleaned_cc100_en_ind_word_switch_oracle/train")
    # merge_datasets("/home/aiops/liuqian/TinyLlama/hf_dataset/cleaned_cc100_word_switch_0.1_new/valid",
    #                "/home/aiops/liuqian/TinyLlama/hf_dataset/redpajama_20b/valid",
    #                "/home/aiops/liuqian/TinyLlama/hf_dataset/cleaned_cc100_en_ind_word_switch_oracle/valid")