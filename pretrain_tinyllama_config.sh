export WANDB_PROJECT=TinyLLama-1B-Llama-Random-LR
export WANDB_ENTITY=SivilTaram
export WANDB_API_KEY=c5e9b1a784400b81d8ce5537a23ee47f6d034783
export MODEL_NAME=tinyllama_1B_config_$1
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
    --data_yaml_file llama_config_lr/$1.yaml \
    --out_name $MODEL_NAME \
    --load_from ../models/TinyLlama-1T-Model/lit_model.pth
    # --load_from /home/aiops/liuqian/TinyLlama/TinyLlama/checkpoints/tinyllama_1T_120M_mistral_en/iter-040000-ckpt.pth
