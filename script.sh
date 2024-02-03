python scripts/convert_hf_checkpoint.py --checkpoint_dir ../TinyLlama-1.5T-Model --model_name tiny_LLaMA_1b

python test_weight.py --checkpoint_dir out/TinyLlama-1.1B-intermediate-900B


python pretrain/tinyllama_code.py --devices 8 --train_data_dir data/code_specialist_python_java_javascript_c_go_8192



python scripts/prepare_starcoder.py --source_path data/starcoderdata/ --tokenizer_path data/llama --destination_path data/code_specialist_python_java_javascript_c_go_8192 --split train --percentage 1.0 --filenames_subset ["python","cpp","go","java","javascript"] --chunk_size 4194816




/data/TinyLlama/out/code_tiny_LLaMA_1b_python_java_go_cpp_javascript/iter-032000-ckpt.pth

python scripts/convert_lit_checkpoint.py --out_dir /home/aiops/liuqian/TinyLlama/models/TinyLlama-v2-800B-Model/ --checkpoint_name step-0475000-ckpt.pth --model_name tiny_LLaMA_1b

python scripts/convert_lit_checkpoint.py --out_dir /data/TinyLlama/out/tiny_LLaMA_1b/ --checkpoint_name iter-100000-ckpt.pth --model_name tiny_LLaMA_1b

python convert_lit_checkpoint.py  --out_dir /home/aiops/liuqian/TinyLlama/models/tinyllama_120M_mistral_en --checkpoint_name iter-020000-ckpt.pth --model_name tiny_LLaMA_mistral_120M

python convert_hf_checkpoint.py --checkpoint_dir /home/aiops/liuqian/TinyLlama/models/tinyllama_120M_mistral_en --model_name tiny_LLaMA_mistral_120M_expand

python convert_hf_checkpoint.py --checkpoint_dir /home/aiops/liuqian/TinyLlama/models/TinyLlama-3T-Model --model_name tiny_LLaMA_1b_expand
