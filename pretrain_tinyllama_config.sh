export WANDB_PROJECT=TinyLLama-Vocab-Expand
export WANDB_ENTITY=SivilTaram
export WANDB_API_KEY=c5e9b1a784400b81d8ce5537a23ee47f6d034783
export MODEL_NAME=tinyllama_1T_1B_model_vocab_old
export WANDB_NAME=$MODEL_NAME
export NUMBER_OF_GPU=8

lightning run model \
    --node-rank=0  \
    --main-address=127.0.0.1 \
    --accelerator=cuda \
    --num-nodes=1 \
    --devices=$NUMBER_OF_GPU \
    pretrain/tinyllama.py --devices $NUMBER_OF_GPU \
    --train_data_dir ../lit_dataset \
    --val_data_dir ../lit_dataset \
    --data_yaml_file vocab_config/vocab_old.yaml \
    --out_name $MODEL_NAME \
    --load_from ../TinyLlama-1T-Model-Expand/lit_model.pth