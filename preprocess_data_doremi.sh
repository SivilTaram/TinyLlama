export DATASET_NAME=qwen2_data_mixture

python scripts/prepare_file_doremi.py --source_path /nfs-share/qwen_mixed_corpus --tokenizer_path data/qwen2 --destination_path ../lit_dataset_qwen_final --short_name $DATASET_NAME --split train
# python scripts/prepare_file_doremi.py --source_path ../hf_dataset/$DATASET_NAME --tokenizer_path data/qwen2 --destination_path ../lit_dataset_qwen_final --short_name $DATASET_NAME --split valid --chunk_size 32776
