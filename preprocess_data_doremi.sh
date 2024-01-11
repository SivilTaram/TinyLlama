export DATASET_NAME=data_mixture

python scripts/prepare_file_doremi.py --source_path ../hf_dataset/$DATASET_NAME --tokenizer_path data/llama --destination_path ../lit_dataset_llama --short_name $DATASET_NAME --split train
python scripts/prepare_file_doremi.py --source_path ../hf_dataset/$DATASET_NAME --tokenizer_path data/llama --destination_path ../lit_dataset_llama --short_name $DATASET_NAME --split valid
