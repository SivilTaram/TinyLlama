export DATASET_NAME=redpajama_20b

python scripts/prepare_file.py --source_path ../hf_dataset/$DATASET_NAME --tokenizer_path data/mistral_sea --destination_path ../lit_dataset --short_name mistral_$DATASET_NAME --split train
python scripts/prepare_file.py --source_path ../hf_dataset/$DATASET_NAME --tokenizer_path data/mistral_sea --destination_path ../lit_dataset --short_name mistral_$DATASET_NAME --split valid

# python scripts/prepare_file_doremi.py --source_path ../hf_dataset/$DATASET_NAME --tokenizer_path data/new_llama --destination_path ../lit_dataset --short_name $DATASET_NAME --split train
# python scripts/prepare_file_doremi.py --source_path ../hf_dataset/$DATASET_NAME --tokenizer_path data/new_llama --destination_path ../lit_dataset --short_name $DATASET_NAME --split valid --chunk_size $((2049 * 512)) 

# export DATASET_NAME=ccaligned_parallel

# python scripts/prepare_file.py --source_path ../hf_dataset/$DATASET_NAME --tokenizer_path data/llama --destination_path ../lit_dataset --short_name $DATASET_NAME --split train
