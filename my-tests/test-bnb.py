import bitsandbytes as bnb
import torch

original_dtype = torch.float16
compute_dtype = None
device = "cuda"
layer_shape = (64, 64)
compress_statistics = True
bias = False
quant_type = "nf4"
quant_storage = original_dtype

linear = torch.nn.Linear(*layer_shape, dtype=original_dtype, device="cpu")  # original layer

# Quantizing original layer
linear_q = bnb.nn.Linear4bit(
    linear.in_features,
    linear.out_features,
    bias=bias,
    compute_dtype=compute_dtype,
    compress_statistics=compress_statistics,
    quant_type=quant_type,
    quant_storage=quant_storage,
    device="meta",
)
new_weight = bnb.nn.Params4bit(data=linear.weight, quant_type=quant_type, quant_storage=quant_storage, compress_statistics=compress_statistics, requires_grad=False)
linear_q.weight = new_weight
if bias:
    linear_q.bias = torch.nn.Parameter(linear.bias)
breakpoint()
linear_q = linear_q.to(device)
