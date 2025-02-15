import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)
from torchviz import make_dot

model = torch.nn.Sequential()
class CheckpointWrapper(torch.nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.model_args = model_args
        self.layers = torch.nn.ModuleList()
        for i in range(model_args.n_layers):
            self.layers.append(TransformerBlock(model_args))

    def forward(self, x):
        for i in range(self.model_args.n_layers):
            if i > 0 and i < self.model_args.n_layers - 1:
                print(f"Checkpoint {i} {self.model_args.n_layers}")
                x = torch.utils.checkpoint.checkpoint(self.layers[i].forward, x)
            else:
                x = self.layers[i].forward(x)
        return x

def main():
    torch.manual_seed(42)
    model_args = ModelArgs(
        n_layers=4,
        vocab_size=128,
        n_heads=4,
        dim=128,
        max_seq_len=128,
        dropout_p=0.0,
    )
    
    # model = torch.nn.Sequential()
    # for i in range(model_args.n_layers):
    #     model.add_module(f"layer_{i}", TransformerBlock(model_args))
    # model.cuda()
    # inp = torch.randn(2, 128, model_args.dim, device="cuda")
    # output = model(inp)
    # with open("model_no_checkpoint.dot", "w") as f:
    #     f.write(make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True).source)

    model = CheckpointWrapper(model_args)
    model.cuda()
    inp = torch.randn(2, 128, model_args.dim, device="cuda")
    output = model(inp)
    with open("model_checkpoint.dot", "w") as f:
        f.write(make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True).source)

    # for i in range(len(model.layers)):
    #     model.layers[i] = torch.utils.checkpoint.checkpoint(model.layers[i])

    # output = model(inp)
    # with open("model_checkpoint.dot", "w") as f:
    #     f.write(make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True).source)

if __name__ == "__main__":
    main()

