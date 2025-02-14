import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

torch.cuda.reset_max_memory_allocated()
torch.cuda.reset_peak_memory_stats()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, cache_dir="./hf_cache", device_map="meta"
)
params = dict(model.named_parameters())
for name, param in params.items():
    print(name, param.shape, param.device)

print(torch.cuda.max_memory_allocated())
print(torch.cuda.memory_summary())

