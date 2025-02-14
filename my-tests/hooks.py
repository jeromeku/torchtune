import torch
from torch.nn.modules.module import register_module_forward_hook


def my_hook(module, input, output):
    print(f"my_hook {module} {input} {output}")

handle = register_module_forward_hook(my_hook)
model = torch.nn.Linear(10, 10)

x = torch.randn(10)
y = model(x)

handle.remove()
