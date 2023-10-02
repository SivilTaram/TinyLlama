export WANDB_API_KEY=c5e9b1a784400b81d8ce5537a23ee47f6d034783
export WANDB_PROJECT=TinyLLama
export WANDB_ENTITY=SivilTaram

lightning run model \
    --node-rank=0  \
    --main-address=127.0.01 \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=1 \
    pretrain/tinyllama.py --devices 8 --train_data_dir /data/tinyllama_madlad_dataset
