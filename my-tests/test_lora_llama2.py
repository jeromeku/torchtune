import sys

sys.path.append(".")
import torch
from torch import nn
from torchao.dtypes.nf4tensor import NF4Tensor

from tests.test_utils import assert_expected, fixed_init_model
from torchtune import training
from torchtune.models.llama2 import llama2, lora_llama2
from torchtune.models.llama2._component_builders import lora_llama2_self_attention
from torchtune.modules.low_precision import FrozenNF4Linear
from torchtune.modules.peft import LoRALinear, get_merged_lora_ckpt
from torchtune.training.seed import set_seed

RANK = 4
ALPHA = 1.0
BSZ = 2
SEQ_LEN = 32
EMBED_DIM = 64
NUM_HEADS = 4
NUM_KV_HEADS = 2
MAX_SEQ_LEN = 64
VOCAB_SIZE = 50
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")
SEED = 16
NUM_LAYERS = 3

def inputs(vocab_size=VOCAB_SIZE):
    return torch.randint(low=0, high=vocab_size, size=(BSZ, SEQ_LEN))


def get_lora_llama2(
    lora_modules,
    apply_lora_to_mlp,
    apply_lora_to_output,
    vocab_size,
    reset_norm=True,
    quantize_base=False,
    embed_dim=EMBED_DIM,
    dtype=None,
    num_layers=NUM_LAYERS,
):

    model = lora_llama2(
        lora_attn_modules=lora_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        embed_dim=embed_dim,
        max_seq_len=MAX_SEQ_LEN,
        lora_rank=RANK,
        lora_alpha=ALPHA,
        quantize_base=quantize_base,
    )
    # To make final outputs less trivial
    if reset_norm:
        model.norm = nn.Identity()

    # dtype=None means to just read dtype from parameters
    # in the model. This dtype is set explicitly to bf16 currently
    # when initializing QLoRA models, as ops such as `arange` aren't
    # yet supported with the actual nf4 tensor dtype yet.
    fixed_init_model(model, dtype=dtype)

    return model


def get_ref_llama2(
    num_layers=NUM_LAYERS,
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_kv_heads=NUM_KV_HEADS,
    max_seq_len=MAX_SEQ_LEN,
):
    model = llama2(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
    )
    return model


def test_forward(
    inputs,
    vocab_size=VOCAB_SIZE,
    lora_modules=["q_proj", "v_proj"],
    apply_lora_to_mlp=True,
    apply_lora_to_output=False,
):
    breakpoint()
    model = get_lora_llama2(
        lora_modules, apply_lora_to_mlp, apply_lora_to_output, vocab_size
    )
    breakpoint()
    actual = model(inputs)
    print(f"actual.shape: {actual.shape}")
    # assert_expected(actual.shape, (BSZ, SEQ_LEN, vocab_size))
    # assert_expected(actual.mean(), expected, atol=1e-4, rtol=1e-6)


def test_qlora_linear_quantize_base():
    model = get_lora_llama2(
        lora_modules=["q_proj", "v_proj", "k_proj", "output_proj"],
        apply_lora_to_mlp=True,
        # quantize_base
        apply_lora_to_output=False,
        vocab_size=50,
        quantize_base=True,
        embed_dim=512,
        dtype=torch.bfloat16,
    )
    for module in model.modules():
        if isinstance(module, LoRALinear):
            assert module._quantize_base


def test_qlora_linear_quantize_base_weights():
    # this test checks that modules that don't have LoRA applied to them
    # have their base weights quantized
    model = get_lora_llama2(
        lora_modules=["q_proj", "v_proj"],
        apply_lora_to_mlp=True,
        # quantize_base
        apply_lora_to_output=False,
        vocab_size=50,
        quantize_base=True,
        embed_dim=512,
        dtype=torch.bfloat16,
    )
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            assert module._quantize_base
        elif name in ["k_proj", "output_proj"]:
            assert isinstance(module, FrozenNF4Linear)
            assert isinstance(module.weight, NF4Tensor)


if __name__ == "__main__":
    inputs = inputs()
    print(f"inputs.shape: {inputs.shape}")
    test_forward(inputs)
