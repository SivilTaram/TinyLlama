export DATASET_NAME=qwen2_data_mixture

python scripts/prepare_file.py --source_path /nfs-share/qwen_mixed_corpus --tokenizer_path data/qwen2 --destination_path ../lit_dataset_qwen_final --short_name $DATASET_NAME --split train
python scripts/prepare_file.py --source_path /nfs-share/qwen_mixed_corpus/slimpajama --tokenizer_path data/qwen2 --destination_path ../lit_dataset_qwen_final --short_name en_$DATASET_NAME --split valid  --chunk_size 524416
python scripts/prepare_file.py --source_path /nfs-share/qwen_mixed_corpus/skywork_zh --tokenizer_path data/qwen2 --destination_path ../lit_dataset_qwen_final --short_name zh_$DATASET_NAME --split valid  --chunk_size 524416
python scripts/prepare_file.py --source_path /nfs-share/qwen_mixed_corpus/indonesian_madlad --tokenizer_path data/qwen2 --destination_path ../lit_dataset_qwen_final --short_name id_$DATASET_NAME --split valid  --chunk_size 524416
python scripts/prepare_file.py --source_path /nfs-share/qwen_mixed_corpus/malay_madlad --tokenizer_path data/qwen2 --destination_path ../lit_dataset_qwen_final --short_name ms_$DATASET_NAME --split valid  --chunk_size 524416
python scripts/prepare_file.py --source_path /nfs-share/qwen_mixed_corpus/vietnamese_madlad --tokenizer_path data/qwen2 --destination_path ../lit_dataset_qwen_final --short_name vi_$DATASET_NAME --split valid  --chunk_size 524416
python scripts/prepare_file.py --source_path /nfs-share/qwen_mixed_corpus/thai_madlad --tokenizer_path data/qwen2 --destination_path ../lit_dataset_qwen_final --short_name th_$DATASET_NAME --split valid  --chunk_size 524416
