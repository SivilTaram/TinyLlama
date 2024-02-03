import os
import glob
from tqdm import tqdm
import sys
import random
import subprocess
import json
from functools import partial
import multiprocessing

# def get_line_count(filename):
#     with open(filename, 'r') as file:
#         line_count = sum(1 for line in file)
#     return line_count

def get_line_count(filename):
    try:
        result = subprocess.run(['wc', '-l', filename], capture_output=True, text=True, check=True)
        line_count = int(result.stdout.strip().split()[0])
        return line_count
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None

def split_file_into_fixed_chunks(read_folder, target_folder, prefix,
                                 chunk_number=500, chunk_threshold=500, oversampling=1.0):
    # take all files with the prefix in the folder
    file_paths = glob.glob(os.path.join(read_folder, prefix + "*"))
    # if the target folder + prefix does not exist, create it
    write_folder = os.path.join(target_folder, prefix)
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)
    else:
        print("The target folder {} already exists".format(write_folder))
        return
    global_index = 1
    total_size = 0
    for file_path in file_paths:
        # get the file size
        file_size = get_line_count(file_path)
        print("File {} has {} lines".format(file_path, file_size))
        total_size += file_size
    # oversampling
    total_size = int(total_size * oversampling)
    # calculate the chunk size
    chunk_size = int(total_size // chunk_number) + 1
    buffer_lines = []
    # get the file paths as
    print("Total size: {} lines".format(total_size))
    print("File paths: {}".format(file_paths))
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf8") as f:
            lines = f.readlines()
            # over sampling the lines
            if oversampling > 1.01:
                oversample_size = int(len(lines) * oversampling)
                # randomly sample the lines to make the number of lines oversampling times
                sampled_lines = lines + random.choices(lines,
                                                       k=oversample_size - len(lines))
                # shuffle the lines
                random.shuffle(sampled_lines)
            else:
                sampled_lines = lines
            for line in tqdm(sampled_lines):
                buffer_lines.append(line)
                if len(buffer_lines) >= chunk_size:
                    print("Writing chunk {} with lines {}".format(global_index, len(buffer_lines)))
                    with open(os.path.join(write_folder, "chunk_{}.jsonl".format(global_index)), "w", encoding="utf8") as wf:
                        for line in buffer_lines:
                            wf.write(line)
                    global_index += 1
                    # break
                    if global_index >= chunk_threshold:
                        break
                    buffer_lines.clear()

        if global_index >= chunk_threshold:
            break
    
    # this is the last chunk
    # assert global_index == chunk_number, "The number of chunks is not equal to the chunk number"
    if global_index != chunk_number:
        print("The number of chunks is not equal to the chunk number")

    with open(os.path.join(write_folder, "chunk_{}.jsonl".format(global_index)), "a+", encoding="utf8") as wf:
        for line in buffer_lines:
            wf.write(line)
    buffer_lines.clear()
    
def merge_chunk_files(read_folder, target_folder, chunk_id, chunk_span=3):
    # for all sub folders in the read folder, read the chunk_id file and merge them
    try:
        print("Processing chunk {}".format(chunk_id))
        sub_folders = os.listdir(read_folder)
        # exclude "train" and "valid"
        sub_folders = [sub_folder for sub_folder in sub_folders if sub_folder not in ["train", "valid"]]
        all_lines = []
        for sub_folder in sub_folders:
            for idx in range(chunk_id, chunk_id + chunk_span):
                # get the chunk file path
                chunk_file_path = os.path.join(read_folder, sub_folder, "chunk_{}.jsonl".format(idx))
                # read the chunk file
                if not os.path.exists(chunk_file_path):
                    print("The chunk file {} does not exist".format(chunk_file_path))
                    continue
                with open(chunk_file_path, "r", encoding="utf8") as f:
                    lines = f.readlines()
                    if "valid" in target_folder:
                        # deduplicate the lines
                        lines = list(set(lines))
                    all_lines.extend(lines)
        # deduplicate the lines and shuffle them
        all_lines = list(set(all_lines))
        random.shuffle(all_lines)
        with open(os.path.join(target_folder, "chunk_{}.jsonl".format(chunk_id)), "w", encoding="utf8") as wf:
            wf.writelines(all_lines)
        print("Finished processing chunk {}".format(chunk_id))
    except Exception as e:
        print("Error: {}".format(e))

def merge_chunk_files_parallel(read_folder, start_id, end_id, chunk_span, prefix):
    # top 480 are training, the rest 20 are validation
    chunk_ids = list(range(start_id, end_id, chunk_span))
    # chunk_ids = list(range(161, 162))
    merge_func = partial(merge_chunk_files,
                         read_folder, 
                         os.path.join(read_folder, prefix), 
                         chunk_span=chunk_span)
    with multiprocessing.Pool(processes=3) as pool:
        pool.map(merge_func, chunk_ids)

if __name__ == "__main__":
    prefix_to_oversampling = {"redpajama": 0.52,
                            "cleaned_cc100_ind_dedup": 1.590717536,
                            "cleaned_cc100_lao_dedup": 1.310873169,
                            "cleaned_cc100_ms_dedup": 2.572692433,
                            "cleaned_cc100_th_dedup": 2.06576487,
                            "cleaned_cc100_vi_dedup": 1.175150307,
                            "ebook_id_non_ocr": 1.784073169,
                            "ebook_id_ocr": 1.990833428,
                            "ebook_ms_non_ocr": 1.720609279,
                            "ebook_th_non_ocr": 1.29915457,
                            "ebook_vi_non_ocr": 1.571884399,
                            "indonesian_madlad": 1.284123889,
                            "malay_madlad": 2.309105402,
                            "subtitle_id": 1.943431342,
                            "subtitle_ms": 1.644883369,
                            "subtitle_th": 2.258092665,
                            "subtitle_vi": 1.538324544,
                            "thai_madlad": 1.207230403,
                            "translation_indonesian": 2.344973337,
                            "translation_thai": 1.810940887,
                            "translation_vietnamese": 1.135346713,
                            "vietnamese_madlad": 0.6959888856,
                            "wikipedia_id_text": 1.955248684,
                            "wikipedia_ms_text": 2.067696885,
                            "wikipedia_th_text": 2.027472096,
                            "wikipedia_vi_text": 1.616933695}

    # for prefix, oversampling in prefix_to_oversampling.items():
    #     print("Processing {}".format(prefix))
    #     split_file_into_fixed_chunks("/nfs-share/sea_corpus",
    #                                 "/nfs-share/mistral_origin_mixed_corpus",
    #                                 prefix,
    #                                 chunk_number=500,
    #                                 chunk_threshold=501,
    #                                 oversampling=oversampling)
    # split_file_into_fixed_chunks("/nfs-share/redpajama_120B_sample",
    #                             "/nfs-share/sea_mixed_corpus",
    #                             "redpajama_120B_sample",
    #                             chunk_number=500,
    #                             oversampling=1.0)
    merge_chunk_files_parallel("/nfs-share/mistral_origin_mixed_corpus",
                               start_id=1,
                               end_id=500,
                               chunk_span=4,
                               prefix="train")
    merge_chunk_files_parallel("/nfs-share/mistral_origin_mixed_corpus",
                               start_id=500,
                               end_id=501,
                               chunk_span=1,
                               prefix="valid")