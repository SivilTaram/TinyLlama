export WANDB_PROJECT=TinyLLama
export WANDB_ENTITY=SivilTaram

lightning run model \
    --node-rank=0  \
    --main-address=127.0.01 \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=1 \
    pretrain/tinyllama.py --devices 8 --train_data_dir /data/tinyllama_madlad_dataset
