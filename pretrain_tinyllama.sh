export WANDB_PROJECT=TinyLLama
export WANDB_ENTITY=SivilTaram
export WANDB_API_KEY=c5e9b1a784400b81d8ce5537a23ee47f6d034783
export MODEL_NAME=tinyllama_1T_en_id_1v1

lightning run model \
    --node-rank=0  \
    --main-address=127.0.01 \
    --accelerator=cuda \
    --num-nodes=1 \
    --devices=8 \
    pretrain/tinyllama.py --devices 8 \
    --train_data_dir ../lit_dataset \
    --val_data_dir ../lit_dataset \
    --out_name $MODEL_NAME \
    --load_from ../TinyLlama-1T-Model/lit_model.pth