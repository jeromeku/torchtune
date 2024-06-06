# tune run --nnodes=1 --nproc_per_node=2 custom_recipes/lora_finetune_fsdp2.py --config custom_configs/7B_lora.yaml
#tune run --nnodes=1 --nproc_per_node=2 custom_recipes/lora_tiny_llama.py --config custom_configs/tiny_llama.yaml
# tune run --nnodes=1 --nproc_per_node=2 lora_finetune_fsdp2_profile --config llama2/7B_lora_profile
tune run lora_finetune_single_device --config llama2/7B_lora_single_device \
profile.enabled=True \
profile.schedule.active=1 \
profile.schedule.wait=1 \
profile.schedule.warmup=1 \
profile.schedule.repeat=1