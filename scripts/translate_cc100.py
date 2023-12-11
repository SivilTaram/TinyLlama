from transformers import pipeline
import os
import json
from tqdm import tqdm

translator = pipeline("translation_id_to_en", 
                      model="facebook/m2m100_418M",
                      device_map="auto")

def translate_file(folder_path, output_folder):
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    # all_files = os.listdir(folder_path)
    # for file_path in all_files:
    output_write_f = open(output_folder, 'w')
    with open(folder_path, 'r') as f:
        lines = f.readlines()
        # take line from "text" in jsonl
        lines = [json.loads(line)['text'] for line in lines]
        # order lines by length
        # print("Processing file: ", file_path)
        # split lines into 1000 lines per batch
        for i in tqdm(range(0, len(lines), 10000)):
            chunk = lines[i:i+ 10000]
            # sort the batch, and record the original index
            sorted_list = sorted(enumerate(chunk), key=lambda x: (len(x[1]), x[0]))
            # separae the index and the text
            sorted_index = [idx for idx, _ in sorted_list]
            sorted_text = [text for _, text in sorted_list]
            output_text = {}
            # tune the batch_size and max_length according to the length of the batch
            for j in range(0, len(sorted_text), 1024):
                batch = sorted_text[j:j+1024]
                batch_indexes = sorted_index[j:j+1024]
                batch_max_length = len(batch[-1])
                batch_size = int(512 / batch_max_length * 36)
                translated_lines = translator(batch, 
                                              batch_size=batch_size,
                                              max_length=int(batch_max_length*1.2))
                for idx, line in zip(batch_indexes, translated_lines):
                    if idx not in output_text:
                        output_text[idx] = line['translation_text']
                    else:
                        print("Duplicate index: ", idx)
            # write both the original and the translated lines
            for j in range(len(chunk)):
                if j in output_text:
                    output_write_f.write(json.dumps({"original_text": chunk[j],
                                                    "translation_text": output_text[j]}) + '\n')
                else:
                    print("Missing index: ", j)
            output_write_f.flush()
    output_write_f.close()
    
if __name__ == "__main__":
    translate_file("../../hf_dataset/cleaned_cc100_head_1m.jsonl",
                   "../../hf_dataset/cleaned_cc100_head_1m_translated_m2m.jsonl")