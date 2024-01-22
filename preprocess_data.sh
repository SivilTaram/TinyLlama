export DATASET_NAME=llama_mixed_sea

# only take 20% because we do not enough disk space
python scripts/prepare_file.py --source_path /nfs-share/mistral_origin_mixed_corpus --tokenizer_path data/new_llama --destination_path ../lit_dataset_new_llama --short_name all_$DATASET_NAME --split train --percentage 0.05
python scripts/prepare_file.py --source_path /nfs-share/mistral_origin_mixed_corpus/redpajama --tokenizer_path data/new_llama --destination_path ../lit_dataset_new_llama --short_name en_$DATASET_NAME --split valid --chunk_size 32784
python scripts/prepare_file.py --source_path /nfs-share/mistral_origin_mixed_corpus/wikipedia_id_text --tokenizer_path data/new_llama --destination_path ../lit_dataset_new_llama --short_name id_$DATASET_NAME --split valid  --chunk_size 32784
python scripts/prepare_file.py --source_path /nfs-share/mistral_origin_mixed_corpus/wikipedia_ms_text --tokenizer_path data/new_llama --destination_path ../lit_dataset_new_llama --short_name ms_$DATASET_NAME --split valid  --chunk_size 32784
python scripts/prepare_file.py --source_path /nfs-share/mistral_origin_mixed_corpus/wikipedia_th_text --tokenizer_path data/new_llama --destination_path ../lit_dataset_new_llama --short_name th_$DATASET_NAME --split valid  --chunk_size 32784
python scripts/prepare_file.py --source_path /nfs-share/mistral_origin_mixed_corpus/wikipedia_vi_text --tokenizer_path data/new_llama --destination_path ../lit_dataset_new_llama --short_name vi_$DATASET_NAME --split valid  --chunk_size 32784