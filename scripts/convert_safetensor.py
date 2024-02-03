from safetensors import SafetensorError
from safetensors.torch import load_file as safetensors_load
from pathlib import Path
import torch
import os

directory = Path("/home/aiops/liuqian/Qwen2-beta-0_5B")

print("Converting .safetensor files to PyTorch binaries (.bin)")
for safetensor_path in directory.glob("*.safetensors"):
    bin_path = safetensor_path.with_suffix(".bin")
    try:
        result = safetensors_load(safetensor_path)
    except SafetensorError as e:
        raise RuntimeError(f"{safetensor_path} is likely corrupted. Please try to re-download it.") from e
    print(f"{safetensor_path} --> {bin_path}")
    torch.save(result, bin_path)
    os.remove(safetensor_path)
