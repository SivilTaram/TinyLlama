export WANDB_PROJECT=Qwen-Model-Training
export WANDB_ENTITY=SivilTaram
export WANDB_API_KEY=c5e9b1a784400b81d8ce5537a23ee47f6d034783
export MODEL_NAME=qwen_500M_debug
export WANDB_NAME=$MODEL_NAME
export NUMBER_OF_GPU=8

lightning run model \
    --node-rank=0  \
    --main-address=127.0.01 \
    --accelerator=cuda \
    --num-nodes=1 \
    --devices=$NUMBER_OF_GPU \
    pretrain/tinyllama.py --devices $NUMBER_OF_GPU \
    --train_data_dir ../lit_dataset_qwen_toy \
    --val_data_dir ../lit_dataset_qwen_toy \
    --out_name $MODEL_NAME \
    --resume True \
    --load_from /home/aiops/liuqian/Qwen2-beta-0_5B/lit_model.pth

    # --data_yaml_file config/llama_en.yaml \
    # --load_from /home/aiops/liuqian/TinyLlama/TinyLlama/checkpoints/tinyllama_120M_long/iter-050000-ckpt.pth
    # --load_from ../TinyLlama-1T-Model/lit_model.pth
    # --load_from /home/aiops/liuqian/TinyLlama/TinyLlama/checkpoints/tinyllama_1T_120M_mistral_en/iter-040000-ckpt.pth
