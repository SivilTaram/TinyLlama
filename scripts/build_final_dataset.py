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
                                 chunk_number=500, chunk_threshold=500):
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
    # calculate the chunk size
    chunk_size = int(total_size // chunk_number) + 1
    buffer_lines = []
    # get the file paths as
    print("Total size: {} lines".format(total_size))
    print("File paths: {}".format(file_paths))
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf8") as f:
            lines = f.readlines()
            for line in tqdm(lines):
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
    
def merge_chunk_files(read_folder, target_folder, chunk_id, prefix_to_oversampling, chunk_span=3):
    # for all sub folders in the read folder, read the chunk_id file and merge them
    try:
        print("Processing chunk {}".format(chunk_id))
        sub_folders = os.listdir(read_folder)
        # remove all folder which is not in the key of prefix_to_oversampling
        sub_folders = [sub_folder for sub_folder in sub_folders if sub_folder in prefix_to_oversampling]
        all_lines = []
        for sub_folder in sub_folders:
            sampling_rate = prefix_to_oversampling[sub_folder]
            print("Processing sub folder {} with sampling rate {}".format(sub_folder, sampling_rate))
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
                    else:
                        # oversampling
                        if sampling_rate > 1.0:
                            lines = random.choices(lines, k=int(len(lines) * sampling_rate))
                        else:
                            lines = random.sample(lines, k=int(len(lines) * sampling_rate))
                    all_lines.extend(lines)

        random.shuffle(all_lines)
        with open(os.path.join(target_folder, "chunk_{}.jsonl".format(chunk_id)), "w", encoding="utf8") as wf:
            wf.writelines(all_lines)
        print("Finished processing chunk {}".format(chunk_id))
    except Exception as e:
        print("Error: {}".format(e))

def merge_chunk_files_parallel(read_folder, start_id, end_id, chunk_span, prefix, prefix_to_oversampling):
    # top 480 are training, the rest 20 are validation
    chunk_ids = list(range(start_id, end_id, chunk_span))
    # chunk_ids = list(range(161, 162))
    merge_func = partial(merge_chunk_files,
                         read_folder, 
                         os.path.join(read_folder, prefix),
                         prefix_to_oversampling=prefix_to_oversampling,
                         chunk_span=chunk_span)
    with multiprocessing.Pool(processes=3) as pool:
        pool.map(merge_func, chunk_ids)

if __name__ == "__main__":
    prefix_to_oversampling = {"slimpajama": 0.42872358913,
                            "skywork_zh": 0.15176751778,
                            "cleaned_cc100_ind_dedup": 0.22456749057,
                            "cleaned_cc100_lao_dedup": 0.28351052782,
                            "cleaned_cc100_ms_dedup": 0.35477340811,
                            "cleaned_cc100_th_dedup": 0.30678568228,
                            "cleaned_cc100_vi_dedup": 0.25018087497,
                            "ebook_id_non_ocr": 0.30010204349,
                            "ebook_id_ocr": 0.29621025740,
                            "ebook_ms_non_ocr": 0.29436935864,
                            "ebook_th_non_ocr": 0.32090138068,
                            "ebook_vi_non_ocr": 0.31698371533,
                            "indonesian_madlad": 0.25349047449,
                            "malay_madlad": 0.44282729044,
                            "subtitle_id": 0.30223081478,
                            "subtitle_ms": 0.29410168250,
                            "subtitle_th": 0.30335005635,
                            "subtitle_vi": 0.28513329196,
                            "thai_madlad": 0.46186715445,
                            "translation_indonesian": 0.31034981959,
                            "translation_thai": 0.32310529624,
                            "translation_vietnamese": 0.34480255283,
                            "vietnamese_madlad": 0.15085510812,
                            "website_pdf_thai": 0.32950514066,
                            "website_thai": 0.28559413464,
                            "wikipedia_id_text": 0.40333269710,
                            "wikipedia_ms_text": 0.40333269710,
                            "wikipedia_th_text": 0.40333269710,
                            "wikipedia_vi_text": 0.40333269710}

    # for prefix, _ in prefix_to_oversampling.items():
    #     print("Processing {}".format(prefix))
    #     split_file_into_fixed_chunks("/nfs-share/sea_corpus",
    #                                 "/nfs-share/qwen_mixed_corpus",
    #                                 prefix,
    #                                 chunk_number=500,
    #                                 chunk_threshold=501)

    merge_chunk_files_parallel("/nfs-share/qwen_mixed_corpus_sample",
                               start_id=1,
                               end_id=11,
                               chunk_span=2,
                               prefix="train_en_50",
                               prefix_to_oversampling=prefix_to_oversampling)
    # merge_chunk_files_parallel("/nfs-share/qwen_mixed_corpus_sample",
    #                            start_id=11,
    #                            end_id=12,
    #                            chunk_span=1,
    #                            prefix="valid",
    #                            prefix_to_oversampling=prefix_to_oversampling)