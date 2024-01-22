export WANDB_PROJECT=TinyLLama-Model-Expand
export WANDB_ENTITY=SivilTaram
export WANDB_API_KEY=c5e9b1a784400b81d8ce5537a23ee47f6d034783
export MODEL_NAME=tinyllama_120M_mistral_30k_$1
export WANDB_NAME=$MODEL_NAME
export NUMBER_OF_GPU=4

lightning run model \
    --node-rank=0  \
    --main-address=127.0.0.1 \
    --accelerator=cuda \
    --num-nodes=1 \
    --devices=$NUMBER_OF_GPU \
    pretrain/tinyllama.py --devices $NUMBER_OF_GPU \
    --train_data_dir ../lit_dataset \
    --val_data_dir ../lit_dataset \
    --data_yaml_file sea_config_expand/$1.yaml \
    --out_name $MODEL_NAME \
    --load_from /home/aiops/liuqian/TinyLlama/models/tinyllama_120M_mistral_en/iter-030000-ckpt.pth
    # --resume True \
    # --load_from ../models/TinyLlama-3T-Model/lit_model_1b_expand.pth
