from transformers import AutoTokenizer
from transformers.utils.logging import set_verbosity_debug

set_verbosity_debug()

MODEL_ID = "meta-llama/Llama-3.2-1B"
breakpoint()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
