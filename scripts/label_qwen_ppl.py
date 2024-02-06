from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
import json
import subprocess


def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        "/home/aiops/liuqian/Qwen2-beta-0_5B",
        # cache_dir="/dev/cache/liuqian/santacoder",
        torch_dtype=torch.bfloat16
    )
    model.to("cuda")
    model.eval()
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        # cache_dir="/dev/cache/liuqian/santacoder",
        model_max_length=3072
    )
    return model, tokenizer


def get_hash(example):
    """Get hash of content field."""
    return {"hash": hash(example["text"])} # can use any hashing function here


def check_uniques(example, uniques):
    """Check if current hash is still in set of unique hashes and remove if true."""
    if example["hash"] in uniques:
        uniques.remove(example["hash"])
        return True
    else:
        return False

def check_gpu_memory():
    try:
        # 执行 nvidia-smi 命令并捕获输出
        result = subprocess.check_output("nvidia-smi", shell=True, text=True)
        
        # 在输出中查找显存信息
        lines = result.split('\n')
        for line in lines:
            if 'Memory' in line and 'MiB / 80' in line:
                return True  # 显卡显存为80GB
    except subprocess.CalledProcessError:
        # 处理命令执行错误
        print("Error executing nvidia-smi command.")
    
    return False  # 未找到匹配的显卡信息

is_80gb_gpu = check_gpu_memory()

def label_model_loss(file_path):
    model, tokenizer = load_model()
    tokenizer.pad_token = tokenizer.eos_token
    output_file = file_path.replace(".json", "_with_loss.json")
    dataset = load_dataset("json",
                           data_files=file_path,
                           split="train")
    dataset = dataset.map(get_hash)
    uniques = set(dataset.unique("hash"))
    dataset = dataset.filter(check_uniques, fn_kwargs={"uniques": uniques})
    print("Dataset size:", len(dataset))
    dataset = dataset.sort("text", reverse=True)
    # quickly estimate the length of the whole dataset
    # preprocess and tokenize the dataset
    batch_size = 20 if is_80gb_gpu else 10
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    loss_value = []
    text_value = []
    with torch.no_grad():
        # batched dataset
        i = 0
        while i < len(dataset):
            print(f"Processing {i} - {i+batch_size} examples...")
            batch = dataset[i:i+batch_size]
            # tokenize
            inputs = tokenizer(batch["text"],
                            return_tensors="pt", 
                            max_length=3072,
                            truncation=True,
                            padding=True).to("cuda")
            # forward
            output = model(**inputs)
            # cacluate loss
            labels = inputs["input_ids"]
            length_mask = inputs["attention_mask"][..., 1:]
            shift_logits = output.logits[..., :-1, :].contiguous()
            batch_size = shift_logits.size(0)
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = loss_fn(shift_logits, shift_labels)
            # reshape as [batch size x seq length]
            loss = loss.view(batch_size, -1)
            loss = loss * length_mask
            # average over the sequence length
            loss_list = []
            for i in range(batch_size):
                loss_single = loss[i].sum() / length_mask[i].sum()
                loss_list.append(loss_single.item())
            loss_value.extend(loss_list)
            # save text
            text_value.extend(batch["text"])
            if len(loss_value) >= 1000:
                print("Writing into the file...")
                # write into the file using a+ mode
                with open(output_file, "a+") as f:
                    for text, loss in zip(text_value, loss_value):
                        f.write(json.dumps({"text": text, "loss": loss},
                                        ensure_ascii=False) + "\n")
                loss_value = []
                text_value = []
            i += batch_size
            # update batch size if necessary using the length of the inputs
            total_capacity = 40_000 if is_80gb_gpu else 20_000
            cur_max_length = inputs.input_ids.size(1)
            batch_size = total_capacity // cur_max_length

if __name__ == "__main__":
    label_model_loss("/home/aiops/liuqian/TinyLlama/hf_dataset/qwen2_data_mixture/train/cleaned_cc100_ms_dedup_chunk_1.jsonl")