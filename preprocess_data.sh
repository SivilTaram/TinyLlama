export DATASET_NAME=redpajama_20b

python scripts/prepare_file.py --source_path ../hf_dataset/$DATASET_NAME --tokenizer_path data/llama --destination_path ../lit_dataset --short_name $DATASET_NAME --split train
python scripts/prepare_file.py --source_path ../hf_dataset/$DATASET_NAME --tokenizer_path data/llama --destination_path ../lit_dataset --short_name $DATASET_NAME --split valid
