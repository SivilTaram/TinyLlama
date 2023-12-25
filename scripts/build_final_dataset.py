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

def split_file_into_fixed_chunks(read_folder, target_folder, prefix, chunk_number=500, oversampling=1.0):
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
                    buffer_lines.clear()
    
    # this is the last chunk
    # assert global_index == chunk_number, "The number of chunks is not equal to the chunk number"
    if global_index != chunk_number:
        print("The number of chunks is not equal to the chunk number")
    with open(os.path.join(write_folder, "chunk_{}.jsonl".format(global_index)), "a+", encoding="utf8") as wf:
        for line in buffer_lines:
            wf.write(line)
    buffer_lines.clear()
    
def merge_chunk_files(read_folder, target_folder, chunk_id):
    # for all sub folders in the read folder, read the chunk_id file and merge them
    print("Processing chunk {}".format(chunk_id))
    sub_folders = os.listdir(read_folder)
    # exclude "train" and "valid"
    sub_folders = [sub_folder for sub_folder in sub_folders if sub_folder not in ["train", "valid"]]
    all_lines = []
    for sub_folder in sub_folders:
        for i in range(1, 4):
            idx = chunk_id * 3 + i
            # get the chunk file path
            chunk_file_path = os.path.join(read_folder, sub_folder, "chunk_{}.jsonl".format(idx))
            # read the chunk file
            if not os.path.exists(chunk_file_path):
                print("The chunk file {} does not exist".format(chunk_file_path))
                continue
            with open(chunk_file_path, "r", encoding="utf8") as f:
                lines = f.readlines()
                all_lines.extend(lines)
    # shuffle the lines
    random.shuffle(all_lines)
    # check if every line is a json parseable line
    # new_lines = []
    # for line in all_lines:
    #     try:
    #         text = json.loads(line)["text"]
    #         new_lines.append(line)
    #     except:
    #         print("Chunk: {} The line {} is not a json parseable line".format(chunk_id, line))
    # write the lines to the target file
    with open(os.path.join(target_folder, "chunk_{}.jsonl".format(chunk_id)), "w", encoding="utf8") as wf:
        wf.writelines(all_lines)
    print("Finished processing chunk {}".format(chunk_id))
    

def merge_chunk_files_parallel(read_folder):
    # top 480 are training, the rest 20 are validation
    # chunk_ids = list(range(0, 160))
    chunk_ids = list(range(161, 162))
    merge_func = partial(merge_chunk_files, read_folder, os.path.join(read_folder, "valid"))
    with multiprocessing.Pool(processes=2) as pool:
        pool.map(merge_func, chunk_ids)

if __name__ == "__main__":
    # prefix_to_oversampling = {"cleaned_cc100_ind_dedup_doc": 1.777029879,
    #                           "cleaned_cc100_lao_dedup_doc": 3.334922596,
    #                         "cleaned_cc100_ms_dedup_doc": 2.873862059,
    #                         "cleaned_cc100_th_dedup_doc": 4,
    #                         "cleaned_cc100_vi_dedup_doc": 2.040602955,
    #                         "ebook_id_non_ocr": 1.99344164,
    #                         "ebook_id_ocr": 2.218786596,
    #                         "ebook_ms_non_ocr": 1.947584244,
    #                         "ebook_th_non_ocr": 2.691466146,
    #                         "ebook_vi_non_ocr": 2.714485951,
    #                         "indonesian_madlad": 1.434624562,
    #                         "indonesian_sft_pretrain": 1.357744334,
    #                         "malay_madlad": 2.580363187,
    #                         "subtitle_id": 2.18274727,
    #                         "subtitle_ms": 1.782301869,
    #                         "subtitle_th": 4,
    #                         "subtitle_vi": 2.717562888,
    #                         "thai_madlad": 2.427850849,
    #                         "thai_sft_pretrain": 1.437790694,
    #                         "translation_indonesian": 2.623684902,
    #                         "translation_thai": 3.644710313,
    #                         "translation_vietnamese": 1.973620278,
    #                         "vietnamese_madlad": 1.208597508,
    #                         "vietnamese_sft_pretrain": 1.3796968,
    #                         "wikipedia_id_text": 2.181537889,
    #                         "wikipedia_ms_text": 2.291357552,
    #                         "wikipedia_th_text": 4,
    #                         "wikipedia_vi_text": 2.807650139}
    # for prefix, oversampling in prefix_to_oversampling.items():
    #     print("Processing {}".format(prefix))
    #     split_file_into_fixed_chunks("/nfs-share/sea_corpus",
    #                                 "/nfs-share/sea_mixed_corpus",
    #                                 prefix,
    #                                 chunk_number=500,
    #                                 oversampling=oversampling)
    # split_file_into_fixed_chunks("/nfs-share/redpajama_120B_sample",
    #                             "/nfs-share/sea_mixed_corpus",
    #                             "redpajama_120B_sample",
    #                             chunk_number=500,
    #                             oversampling=1.0)
    merge_chunk_files_parallel("/nfs-share/sea_mixed_corpus")