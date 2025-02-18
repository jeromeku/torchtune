from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
dataset = load_dataset("Open-Orca/SlimOrca-Dedup")
def convert_to_openai_format(conversation):
    new_conv = []
    for conv in conversation:
        new_conv.append({'role': conv['from'], 'content': conv['value']})
    return {'conversations': new_conv}

dataset = dataset['train'].map(lambda x: convert_to_openai_format(x['conversations']))
print(dataset[0])
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
breakpoint()
