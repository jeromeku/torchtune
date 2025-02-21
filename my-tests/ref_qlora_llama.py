import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

HF_HOME = ROOT_DIR / "hf_cache"
assert HF_HOME.exists()

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,"\
    "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["WANDB_PROJECT"] = "qlora-fsdp2"

import datasets.utils.logging as ds_logging
import torch
import transformers.utils.logging as tf_logging
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

USE_QLORA = False
RUN_NAME = "hf-qlora-ref" if USE_QLORA else "hf-lora-ref"

# tf_logging.set_verbosity_debug()
# ds_logging.set_verbosity_debug()

max_seq_length = 2048
torch.set_default_dtype(torch.bfloat16)
model_name = "meta-llama/Llama-3.2-1B-Instruct"
dtype = torch.bfloat16

if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = dtype,
    )
else:
    bnb_config = None

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map = "auto",
    attn_implementation = "sdpa",
    quantization_config = bnb_config,
    torch_dtype = dtype,
)

model.model.layers = model.model.layers[:1]
model.config.num_hidden_layers = 1

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"
# add pad token
added_vocab = tokenizer.get_added_vocab()
pad_token = [w for w in added_vocab if 'pad' in w]
assert len(pad_token) == 1
tokenizer.pad_token = pad_token[0]

lora_config = LoraConfig(
    r = 64,
    lora_alpha = 128,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_dropout = 0,
    bias = "none",
    task_type = TaskType.CAUSAL_LM,
)

# for name, param in model.named_parameters():
#     print(f"{name} {type(param)} {param.dtype} {param.requires_grad}")

# print(" ---------------- ")
# Get LoRA and setup model

# model = get_peft_model(model, lora_config)
# for name, param in model.named_parameters():
#     print(f"{name} {type(param)} {param.requires_grad}")
# breakpoint()
#model = prepare_model_for_kbit_training(model)  

# with torch.no_grad():
#     for name, param in model.named_parameters():
#         if ".lora_A." in name or ".lora_B." in name: assert param.requires_grad
#         else: assert not param.requires_grad

# model.gradient_checkpointing_enable()
# model.enable_input_require_grads()

#print(model)
# for name, param in model.named_parameters():
#     print(f"{name} {type(param)} {param.dtype} {param.requires_grad}")
# # Get dataset
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
# dataset = load_dataset("json", data_files = {"train" : url}, split = "train[:10%]")
dataset = load_dataset("philschmid/dolly-15k-oai-style", split="train", cache_dir=HF_HOME)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    processing_class = tokenizer,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 1,
        max_steps = 10,
        logging_steps = 1,
        output_dir = "outputs",
        seed = 3407,
        max_seq_length = max_seq_length,
        fp16 = model.get_input_embeddings().weight.dtype == torch.float16,
        bf16 = model.get_input_embeddings().weight.dtype == torch.bfloat16,
        report_to = "wandb", # For W&B
        dataset_num_proc = 4,
        run_name = RUN_NAME,
    ),
)
breakpoint()
#trainer.train()
