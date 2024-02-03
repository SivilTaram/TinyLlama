import os

ROOT_DIR = "/nfs-share/mistral_origin_mixed_corpus"
TARGET_DIR = "/home/aiops/liuqian/TinyLlama/hf_dataset/qwen_data_mixture"

def move_file_to_hf_dataset(dataset_folder):
    for sub_dir in os.listdir(dataset_folder):
        if sub_dir in ["train", "valid"]:
            continue
        sub_dir_path = os.path.join(dataset_folder, sub_dir)
        if os.path.isdir(sub_dir_path):
            # take the first 10 chunks and move them to the dataset folder
            for chunk_index in range(1, 11):
                chunk_path = os.path.join(sub_dir_path, f"chunk_{chunk_index}.jsonl")
                if os.path.exists(chunk_path):
                    # read the file and deduplicate it
                    with open(chunk_path, "r") as f:
                        lines = f.readlines()
                        lines = list(set(lines))
                    new_path = os.path.join(TARGET_DIR, "train", f"{sub_dir}_{chunk_index}.jsonl")
                    # write the deduplicated file
                    with open(new_path, "w") as f:
                        f.writelines(lines)
            # use the chunk 11 as validation
            chunk_path = os.path.join(sub_dir_path, f"chunk_11.jsonl")
            if os.path.exists(chunk_path):
                # read the file and deduplicate it
                with open(chunk_path, "r") as f:
                    lines = f.readlines()
                    lines = list(set(lines))
                new_path = os.path.join(TARGET_DIR, "valid", f"{sub_dir}.jsonl")
                # write the deduplicated file
                with open(new_path, "w") as f:
                    f.writelines(lines)

def merge_dataset_and_re_split(dataset_folder):
    sub_dirs = os.listdir(dataset_folder)
    data_dist = { "cleaned_cc100_ind_dedup": 1.590717536,
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
    prefix_set = set(data_dist.keys())

if __name__ == "__main__":
    move_file_to_hf_dataset(ROOT_DIR)