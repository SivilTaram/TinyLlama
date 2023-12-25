from tqdm import tqdm
import json
import random

CC100_PATHS = [
    # "/sail-data/sealm/cleaned_data/cc100-ind/cleaned_cc100_ind_dedup.jsonl",
    "/nfs-share/sea_corpus/cleaned_cc100_lao_dedup.jsonl",
    "/nfs-share/sea_corpus/cleaned_cc100_ms_dedup.jsonl"
]

def merge_documents_with_sliding_window(read_file_path, write_file_path):
    wf = open(write_file_path, "w", encoding="utf8")
    with open(read_file_path, "r", encoding="utf8") as f:
        line_buffer = []
        for line in tqdm(f):
            text = json.loads(line)["text"]
            line_buffer.append(text)
            window_number = random.randint(5, 10)
            if len(line_buffer) >= window_number:
                wf.write(json.dumps({"text": "\n".join(line_buffer)}, ensure_ascii=False) + "\n")
                line_buffer.clear()
        if len(line_buffer) > 0:
            wf.write(json.dumps({"text": "\n".join(line_buffer)}, ensure_ascii=False) + "\n")
            line_buffer.clear()
    wf.close()

if __name__ == "__main__":
    for cc100_path in CC100_PATHS:
        print("Processing {}".format(cc100_path))
        merge_documents_with_sliding_window(cc100_path,
                                            cc100_path.replace(".jsonl", "_doc.jsonl"))