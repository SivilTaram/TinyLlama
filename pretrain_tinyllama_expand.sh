export WANDB_PROJECT=TinyLLama-Model-Long-80B
export WANDB_ENTITY=SivilTaram
export WANDB_API_KEY=c5e9b1a784400b81d8ce5537a23ee47f6d034783
export MODEL_NAME=qwen_500M_mixcorpus_reset_large_batch_5e-2
export WANDB_NAME=$MODEL_NAME
export NUMBER_OF_GPU=8

lightning run model \
    --node-rank=0  \
    --main-address=127.0.0.1 \
    --accelerator=cuda \
    --num-nodes=1 \
    --devices=$NUMBER_OF_GPU \
    pretrain/tinyllama.py --devices $NUMBER_OF_GPU \
    --train_data_dir ../lit_dataset_llama \
    --val_data_dir ../lit_dataset_llama \
    --data_yaml_file config/llama_mix.yaml \
    --out_name $MODEL_NAME \
    --resume True \
    --load_from /home/aiops/liuqian/TinyLlama/models/TinyLlama-v2-800B-Model/step-0475000-ckpt.pth
    # --load_from /home/aiops/liuqian/TinyLlama/models/tinyllama_120M_mistral_en/iter-030000-ckpt.pth
    # --load_from /home/aiops/liuqian/TinyLlama/TinyLlama/checkpoints/tinyllama_1T_120M_mistral_en/iter-040000-ckpt.pth
