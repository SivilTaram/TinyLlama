import os
import random

def get_prefix(folder_path):
    # read all files in the folder
    all_files = os.listdir(folder_path)
    # shuffle
    random.shuffle(all_files)
    # filter prefix with doremi
    all_files = [file_path for file_path in all_files if "train_the_pile_subsample_p10_arxiv" in file_path]
    # print first 100
    print(len(all_files))
    
    
def delete_prefix(folder_path, prefix):
    # read all files in the folder
    all_files = os.listdir(folder_path)
    # delete prefix
    for file_path in all_files:
        if file_path.startswith(prefix):
            print("Deleting file: ", file_path)
            os.remove(os.path.join(folder_path, file_path))
            
def replace_symbol(folder_path):
    all_files = os.listdir(folder_path)
    for file_path in all_files:
        if file_path.endswith(".jsonl"):
            print("Replacing file: ", file_path)
            with open(os.path.join(folder_path, file_path), "r", encoding="utf8") as f:
                lines = f.readlines()
            with open(os.path.join(folder_path, file_path), "w", encoding="utf8") as f:
                for line in lines:
                    line = line.replace(r"\\n", r"\n")
                    f.write(line)
            
                    
if __name__ == "__main__":
    # get_prefix("../../../TinyLlama-Data/lit_dataset_newtoken")
    delete_prefix("../../../TinyLlama-Data/lit_dataset_newtoken", "train_the_pile_subsample_p10")
    # replace_symbol("../../hf_dataset/madlad_dedup_clean_1/train")
    # replace_symbol("../../hf_dataset/madlad_dedup_clean_1/valid")