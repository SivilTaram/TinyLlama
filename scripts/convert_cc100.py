import json

read_file = "/data/hf_dataset/cc100-merge.txt"
write_file = "/data/hf_dataset/cc100-merge.jsonl"

with open(read_file, "r") as f:
    idx = 0
    while True:
        all_content = f.readlines(10_0000)
        if not all_content:
            break
        idx +=1 
        if idx % 100 == 0:
            print("idx:", idx)
        # split all_content into chunks using \n\n
        chunks = "".join(all_content).split("\n\n")
        # remove chunks which are less than 5 words
        chunks = [chunk for chunk in chunks if len(chunk.split()) > 5]
        # write chunks to write_file
        with open(write_file, "a") as write_f:
            for chunk in chunks:
                write_f.write(json.dumps({"text": chunk}) + "\n")
