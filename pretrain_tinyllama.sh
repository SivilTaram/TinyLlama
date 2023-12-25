export WANDB_PROJECT=TinyLLama
export WANDB_ENTITY=SivilTaram
export WANDB_API_KEY=c5e9b1a784400b81d8ce5537a23ee47f6d034783
export MODEL_NAME=cleaned_cc100_en_ind_word_switch_oracle

lightning run model \
    --node-rank=0  \
    --main-address=127.0.01 \
    --accelerator=cuda \
    --num-nodes=1 \
    --devices=8 \
    pretrain/tinyllama_origin.py --devices 8 \
    --train_data_dir ../lit_dataset \
    --val_data_dir ../lit_dataset \
    --data_yaml_file config/code_switch_orcale.yaml \
    --out_name $MODEL_NAME \
    --resume True \
    --load_from ../TinyLlama-1T-Model/lit_model.pth