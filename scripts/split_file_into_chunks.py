import random
from tqdm import tqdm
from random import shuffle
import os
import json

def split_file_to_chunks(file_path, target_folder):
    chunk_idx = 0
    lines = open(file_path, "r", encoding="utf8").readlines()
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        os.makedirs(f"{target_folder}/train")
        os.makedirs(f"{target_folder}/valid")
    shuffle(lines)
    # split into train / valid
    border = int(len(lines) * 0.95)
    for i in tqdm(range(0, len(lines), 1000000)):
        chunk = lines[i:i+1000000]
        prefix = "train" if i < border else "valid"
        with open(f"{target_folder}/{prefix}/chunk_{chunk_idx}.jsonl", "w", encoding="utf8") as f:
            f.writelines(chunk)
        chunk_idx += 1

def split_train_dev():
    all_file_paths = os.listdir("../../hf_dataset/data_mixture")
    # create the train and dev folder
    if not os.path.exists("../../hf_dataset/data_mixture/train"):
        os.makedirs("../../hf_dataset/data_mixture/train")
        os.makedirs("../../hf_dataset/data_mixture/valid")
    for file_path in all_file_paths:
        if file_path.endswith(".jsonl"):
            print("Processing file: ", file_path)
            # split into train / valid
            try:
                lines = open(f"../../hf_dataset/data_mixture/{file_path}", "r", 
                             encoding="utf8", errors='replace').readlines()
                shuffle(lines)
                border = int(len(lines) * 0.95)
                train_lines = lines[:border]
                valid_lines = lines[border:]
                with open(f"../../hf_dataset/data_mixture/train/{file_path}", "w", encoding="utf8") as f:
                    f.writelines(train_lines)
                with open(f"../../hf_dataset/data_mixture/valid/{file_path}", "w", encoding="utf8") as f:
                    f.writelines(valid_lines)
                os.remove(f"../../hf_dataset/data_mixture/{file_path}")
            except Exception as e:
                print("Error: ", e)
                # delete the original file
                
def fix_files(folder):
    for file_path in os.listdir(folder):
        if file_path.endswith(".jsonl"):
            print("Processing file: ", file_path)
            # split into train / valid
            lines = open(f"{folder}/{file_path}", "r", 
                            encoding="utf8", errors='replace').readlines()
            # fix the lines
            new_lines = []
            for line in lines:
                try:
                    json.loads(line)
                    new_lines.append(line)
                except:
                    print("Error: ", line)
            # with open(f"{folder}/{file_path}", "w", encoding="utf8") as f:
            #     f.writelines(new_lines)

if __name__ == "__main__":
    # split_file_to_chunks("/data/hf_dataset/en_merge_sample.jsonl",
    #                      "/data/hf_dataset/en_merge_sample")
    # split_file_to_chunks("../../hf_dataset/madlad_dedup_clean_1.jsonl",
    #                     "/home/aiops/liuqian/TinyLlama/hf_dataset/madlad_dedup_clean_1")
    # split_train_dev()
    fix_files("../../hf_dataset/data_mixture/train")
