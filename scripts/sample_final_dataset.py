import os
import glob
from tqdm import tqdm
import sys
import random
import subprocess
import json
from functools import partial
import multiprocessing

def get_line_count(filename):
    try:
        result = subprocess.run(['wc', '-l', filename], capture_output=True, text=True, check=True)
        line_count = int(result.stdout.strip().split()[0])
        return line_count
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None

def sample_file_into_chunks(read_folder, 
                            target_folder, 
                            prefix,
                            sample_ratio=0.05,
                            maxium_char=400_000_000):
    # take all files with the prefix in the folder
    file_paths = glob.glob(os.path.join(read_folder, prefix + "*"))
    file_to_count_size = {}
    # if the target folder + prefix does not exist, create it
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    global_index = 1
    total_size = 0
    for file_path in file_paths:
        # get the file size
        file_size = get_line_count(file_path)
        file_to_count_size[file_path] = file_size
        print("File {} has {} lines".format(file_path, file_size))
        total_size += file_size

    buffer_lines = []
    buffer_size = 0
    # get the file paths as
    print("Total size: {} lines".format(total_size))
    
    # if total size is too small, increase the sample ratio by a factor
    if total_size < 100_000:
        sample_ratio = min(1.0, 0.2 * 100_000 / total_size)

    print("File paths: {}".format(file_paths))
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf8") as f:
            lines = f.readlines()
            # sampe lines
            sample_lines = random.sample(lines, int(len(lines) * sample_ratio))
            for line in tqdm(sample_lines):
                buffer_lines.append(line)
                buffer_size += len(line)
                if buffer_size >= maxium_char:
                    print("Writing chunk {} with lines {}".format(global_index, len(buffer_lines)))
                    with open(os.path.join(target_folder, "{}_chunk_{}.jsonl".format(prefix, global_index)), "w", encoding="utf8") as wf:
                        for line in buffer_lines:
                            wf.write(line)
                    global_index += 1
                    buffer_lines.clear()
                    buffer_size = 0

    with open(os.path.join(target_folder, "{}_chunk_{}.jsonl".format(prefix, global_index)), "a+", encoding="utf8") as wf:
        for line in buffer_lines:
            wf.write(line)
    buffer_lines.clear()
    
if __name__ == "__main__":
    prefix_to_use = ["slimpajama",
                     "cleaned_cc100_ind_dedup",
                     "cleaned_cc100_lao_dedup",
                     "cleaned_cc100_ms_dedup",
                     "cleaned_cc100_th_dedup",
                     "cleaned_cc100_vi_dedup",
                     "ebook_id_non_ocr",
                     "ebook_id_ocr",
                     "ebook_ms_non_ocr",
                     "ebook_th_non_ocr",
                     "ebook_vi_non_ocr",
                     "indonesian_madlad",
                     "malay_madlad",
                     "subtitle_id",
                     "subtitle_ms",
                     "subtitle_th",
                     "subtitle_vi",
                     "thai_madlad",
                     "website_pdf_thai",
                     "website_thai",
                     "translation_indonesian",
                     "translation_thai",
                     "translation_vietnamese",
                     "vietnamese_madlad",
                     "wikipedia_id_text",
                     "wikipedia_ms_text",
                     "wikipedia_th_text",
                     "wikipedia_vi_text"]
    for prefix in prefix_to_use:
        print("Sampling prefix {}".format(prefix))
        if prefix == "slimpajama":
            sample_file_into_chunks("/nfs-share/sea_corpus", 
                                    "/home/aiops/liuqian/TinyLlama/hf_dataset/qwen_data_mixture",
                                    prefix, 
                                    sample_ratio=0.012,
                                    maxium_char=400_000_000)
        else:
            sample_file_into_chunks("/nfs-share/sea_corpus", 
                                    "/home/aiops/liuqian/TinyLlama/hf_dataset/qwen_data_mixture",
                                    prefix, 
                                    sample_ratio=0.05,
                                    maxium_char=400_000_000)