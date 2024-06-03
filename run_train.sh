export CHECKPOINT_DIR=/home/ubuntu/model_checkpoints/Llama-2-7b-hf

tune run --nnodes 1 --nproc_per_node 2 lora_finetune_distributed --config custom_configs/7B_lora.yaml checkpointer.checkpoint_dir=${CHECKPOINT_DIR}
