from tqdm import tqdm
import json

def merge_documents_with_sliding_window(read_file_path, write_file_path):
    wf = open(write_file_path, "w", encoding="utf8")
    with open(read_file_path, "r", encoding="utf8") as f:
        lines = f.readlines()
        # every 10 lines, we merge them into one line
        for i in tqdm(range(0, len(lines), 10)):
            chunk = [json.loads(line)["text"] for line in lines[i:i+10]]
            wf.write(json.dumps({"text": "\n".join(chunk)}) + "\n")
    wf.close()
    
if __name__ == "__main__":
    merge_documents_with_sliding_window("../../hf_dataset/cleaned_cc100_ind.jsonl", 
                                        "../../hf_dataset/cleaned_cc100_ind_doc.jsonl")