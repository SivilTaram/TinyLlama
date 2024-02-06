import json
from tqdm import tqdm

def process_eos_text(text):
    processed_text = ""
    sentences = text.split('\n')
    for i, sentence in enumerate(sentences):
        has_period_space = '. ' in sentence
        next_has_period_space = i < len(sentences) - 1 and '. ' in sentences[i + 1]
        if has_period_space or next_has_period_space:
            separator = '\n\n'
        else:
            separator = '\n'
        processed_text += sentence + separator
    return processed_text.rstrip('\n')


def replace_maylay_file():
    maylay_path = "/nfs-share/sea_corpus/malay_madlad.jsonl"
    write_path = "/nfs-share/sea_corpus/malay_madlad_paragraph.jsonl"
    with open(maylay_path, "r", encoding="utf8") as rf:
        buffer_lines = []
        for line in tqdm(rf):
            data = json.loads(line)
            text = data["text"]
            processed_text = process_eos_text(text)
            data["text"] = processed_text
            buffer_lines.append(json.dumps(data, ensure_ascii=False))
            if len(buffer_lines) >= 10000:
                with open(write_path, "a", encoding="utf8") as wf:
                    for line in buffer_lines:
                        wf.write(line + '\n')
                buffer_lines.clear()
    # write the remaining lines
    with open(write_path, "a", encoding="utf8") as wf:
        for line in buffer_lines:
            wf.write(line + '\n')

if __name__ == "__main__":
    replace_maylay_file()