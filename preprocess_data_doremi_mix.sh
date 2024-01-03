export DATASET_NAME=doremi_sample

# python scripts/prepare_file_doremi_mix.py --source_path ../hf_dataset/$DATASET_NAME --old_tokenizer_path data/llama  --new_tokenizer_path data/new_llama --destination_path ../lit_dataset --short_name mix_$DATASET_NAME --split train

python scripts/prepare_file_doremi_mix.py --source_path ../hf_dataset/$DATASET_NAME --old_tokenizer_path data/llama --new_tokenizer_path data/llama --destination_path ../lit_dataset --short_name old_$DATASET_NAME --split valid
