import math

import bitsandbytes as bnb
import torch
import torch.nn as nn
from torchao.dtypes.nf4tensor import NF4Tensor, linear_nf4, to_nf4

from torchtune.modules import FeedForward
from torchtune.modules.low_precision import FrozenNF4Linear
from torchtune.modules.peft import LoRALinear
from torchtune.training.seed import set_seed

SEED = 42
DIM = 512
set_seed(SEED)
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

def scale_hidden_dim_for_mlp(dim: int, multiple_of: int = 256) -> int:
    """Scale hidden dimension for MLP to keep number of parameters and computation constant.

    Args:
        dim (int): Input dimension.
        multiple_of (int): Round scaled dimension to nearest multiple of `multiple_of` for clean computation.

    Returns:
        Scaled hidden dimension.
    """
    # Scale hidden dimension by (2/3)4d for SwiGLU to keep number of
    # parameters and computation constant
    hidden_dim = 4 * int(2 * dim / 3)
    # Round hidden dimension to nearest multiple of `multiple_of`
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim



def _lora_a_init_params(x: nn.Linear) -> None:
    """
    Initialize LoRA A weight to Kaiming uniform.
    """
    nn.init.kaiming_uniform_(x.weight, a=math.sqrt(5))


def _lora_b_init_params(x: nn.Linear) -> None:
    """
    Initialize LoRA B weight to zeros.
    """
    nn.init.zeros_(x.weight)

class LoRALinear(nn.Module):
    """LoRA linear layer as introduced in `LoRA: Low-Rank Adaptation of Large Language Models <https://arxiv.org/abs/2106.09685>`_.

    LoRA perturbs a given layer via a low-rank approximation where only
    the rank decomposition matrices are trainable. In a linear layer instead of
    :math:`x \\mapsto W_0x` a LoRALinear layer is defined as
    :math:`x \\mapsto W_0x + (\\alpha / r)BAx`, where :math:`r` is the rank of
    the matrices :math:`A` and :math:`B` and :math:`\\alpha` is a scaling factor.
    As in the original implementation, we support dropout before multiplication
    by the low-rank matrices.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        rank (int): rank of the low-rank approximation
        alpha (float): scaling factor for the low-rank approximation
        dropout (float): dropout probability. Default: 0.0
        use_bias (bool): whether to include bias in the original linear layer.
            Default: False
        quantize_base (bool): Whether to quantize base linear weight or not.
            Default: False
        **quantization_kwargs: Keyword arguments to pass to `to_nf4` when quantizing the base linear weight.
            Examples of valid arguments are `block_size` and `scaler_block_size`, which control the granularity of
            weight quantization and scaler quantization respectively. This is only used if `quantize_base` is True.
            Default None

    Raises:
        ValueError: If ``quantize_base`` is False, but quantization kwargs are provided.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_bias: bool = False,
        quantize_base: bool = True,
        quant_type: str = "nf4",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        **quantization_kwargs,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.alpha = alpha
        self.use_bias = use_bias
        self._quantize_base = quantize_base
        self.quant_type = quant_type

        if not self._quantize_base and any([v for v in quantization_kwargs.values()]):
            raise ValueError(
                f"``quantize_base`` is False, but received the following quantization arguments: {quantization_kwargs}"
            )

        if self._quantize_base:
            if quant_type == "nf4":
                self.linear = FrozenNF4Linear(in_dim, out_dim, bias=use_bias, device=device, dtype=dtype, **quantization_kwargs)
            elif quant_type == "bnb":
                self.linear = bnb.nn.LinearNF4(in_dim, out_dim, bias=use_bias, compute_dtype=dtype, quant_storage=dtype, device=device, **quantization_kwargs)
            else:
                raise ValueError(f"Unsupported quant_type: {quant_type}")
        else:
            self.linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=self.use_bias, device=device, dtype=dtype)

        # 'self.disabled' is a flag showing whether to turn off LoRA adapters,
        # this can be used in DPO for treating the lora adapters as the policy model
        # and disabling it to treat the base model as the reference model
        self.disabled = False
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.lora_a = nn.Linear(in_features=in_dim, out_features=rank, bias=False)
        self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=False)
        self.merged = False
        self.initialize_parameters()

    def to_empty(
        self, *, device="cuda", recurse: bool = True
    ):
        self.lora_a.to_empty(device=device, recurse=recurse)
        self.lora_b.to_empty(device=device, recurse=recurse)

    def initialize_parameters(self):
        # Initialize as in
        # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L119
        _lora_a_init_params(self.lora_a)
        _lora_b_init_params(self.lora_b)

    def adapter_params(self):
        """
        Return a list of strings corresponding to the names of the ``nn.Parameter`` s in
        the model coming from the adapter.

        For LoRA this means lora_a.weight and lora_b.weight.
        """
        # NOTE: this function has to be updated if the names of "lora_a" and "lora_b"
        # in this module change.
        adapter_params = ["lora_a.weight", "lora_b.weight"]
        return adapter_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape ``(..., in_dim)``

        Returns:
            torch.Tensor: output tensor with shape ``(..., out_dim)``

        """
        out = self.linear(x)
        if self.disabled:
            return out
        lora_out = self.lora_a(self.dropout(x))
        lora_out = (self.alpha / self.rank) * self.lora_b(lora_out)
        return out + lora_out

def lora_llama2_mlp(
    *,
    dim: int,
    hidden_dim: int,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    quantize_base: bool = True,
    adapter_cls = LoRALinear,
) -> FeedForward:
    gate_proj = adapter_cls(
        in_dim=dim,
        out_dim=hidden_dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        quantize_base=quantize_base,
    )
    down_proj = adapter_cls(
        in_dim=hidden_dim,
        out_dim=dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        quantize_base=quantize_base,
    )
    up_proj = adapter_cls(
        in_dim=dim,
        out_dim=hidden_dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        quantize_base=quantize_base,
    )
    return FeedForward(
        gate_proj=gate_proj,
        down_proj=down_proj,
        up_proj=up_proj,
    )

def test_nf4_reconstruction_vs_bnb(dim=DIM, dtype=torch.bfloat16):
    """
    Ensures a BNB NF4 linear and our FrozenNF4Linear have low error when
    reconstructing the respective original weights.
    """

    nf4_linear = FrozenNF4Linear(dim, dim, device="cuda", dtype=dtype)
    orig_weight = nf4_linear.weight.get_original_weight().clone().detach()

    param = bnb.nn.Params4bit(orig_weight, requires_grad=False, quant_type="nf4")
    bnb_nf4_linear = bnb.nn.LinearNF4(
        orig_weight.size(0), orig_weight.size(1), bias=False
    )
    bnb_nf4_linear.weight = param
    bnb_nf4_linear.cuda()

    # From https://github.com/drisspg/transformer_nuggets/blob/f05afad68ad9086d342268f46a7f344617a02314/test/test_qlora.py#L65
    bnb_reconstruction = bnb_nf4_linear(
        torch.eye(dim, dim, dtype=dtype, device="cuda")
    )
    # Ensure nf4_linear and bnb reconstructions are close to each other.
    assert torch.allclose(
        bnb_reconstruction.T, nf4_linear.weight.get_original_weight(), 1e-2
    )

def test_nf4_bnb_linear(dim=DIM, dtype=torch.bfloat16):
    """
    This test ensures that nf4_linear is "no worse" than BNB by ensuring the
    error compared to a bf16 linear is not more than BNB's implementation.
    """

    nf4_linear = FrozenNF4Linear(dim, dim, device="cuda", dtype=dtype)
    orig_weight = nf4_linear.weight.get_original_weight().clone().detach()

    param = bnb.nn.Params4bit(orig_weight, requires_grad=False, quant_type="nf4")
    bnb_nf4_linear = bnb.nn.LinearNF4(
        orig_weight.size(0), orig_weight.size(1), bias=False
    )
    bnb_nf4_linear.weight = param
    bnb_nf4_linear.cuda()

    bf16_linear = torch.nn.Linear(dim, dim, device="cuda", dtype=dtype)

    inp = torch.randn(2, dim, dtype=dtype, device="cuda")

    out_nf4 = nf4_linear(inp)
    out_bnb = bnb_nf4_linear(inp)
    out_ref = bf16_linear(inp)

    err_bnb = out_bnb - out_ref
    err_native = out_nf4 - out_ref
    assert torch.allclose(err_bnb, err_native, 1.0e-2, 1.0e-2)


def test_lora_llama2_mlp():
    hidden_dim = scale_hidden_dim_for_mlp(DIM)
    lora_mlp = lora_llama2_mlp(dim=DIM, hidden_dim=hidden_dim, lora_rank=RANK, lora_alpha=ALPHA)
    print(f"lora_mlp {lora_mlp}")

if __name__ == "__main__":
    test_lora_llama2_mlp()
