export DATASET_NAME=madlad_400_id_clean

python scripts/prepare_file.py --source_path ../hf_dataset/$DATASET_NAME --tokenizer_path data/llama --destination_path ../lit_dataset --short_name $DATASET_NAME --split train
python scripts/prepare_file.py --source_path ../hf_dataset/$DATASET_NAME --tokenizer_path data/llama --destination_path ../lit_dataset --short_name $DATASET_NAME --split valid

# export DATASET_NAME=ccaligned_parallel

# python scripts/prepare_file.py --source_path ../hf_dataset/$DATASET_NAME --tokenizer_path data/llama --destination_path ../lit_dataset --short_name $DATASET_NAME --split train