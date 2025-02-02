import copy
import datetime
import time

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.testing._internal.common_fsdp import MLP

from torchtune import utils
from torchtune.modules.peft import LoRALinear

N_LAYERS = 3
IN_DIM = 5
OUT_DIM = 10
VOCAB_SIZE = 50
NUM_HEADS = 4
NUM_KV_HEADS = 2
EMBED_DIM = 64
MAX_SEQ_LEN = 64


def _get_n_lora_and_tformer_layers(model):
    num_lora_ab = 0
    num_transformer_layers = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            num_nested_linears = len(
                [m for m in module.modules() if isinstance(m, nn.Linear)]
            )
            num_lora_ab += num_nested_linears
        if isinstance(module, TransformerDecoderLayer):
            num_transformer_layers += 1

    return num_lora_ab, num_transformer_layers





def dist_print(*msg, delay=.2):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        time.sleep(rank * delay)
        print(f"[rank{rank}]", *msg, flush=True)
    else:
        print(*msg, flush=True)

def dist_breakpoint():
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            import pdb; pdb.set_trace()
        torch.distributed.barrier()
    else:
        import pdb; pdb.set_trace()

def broadcast_full_state_dict(full_sd):
    result = []
    if torch.distributed.get_rank() == 0:
        result.append(full_sd)
    else:
        result.append(None)
    torch.distributed.broadcast_object_list(result, src=0)
    return result[0]

def test_lora_state_dict():
    rank = torch.distributed.get_rank()
    is_rank_zero = rank == 0
    mlp_dim = 4
    epochs = 5
    torch.manual_seed(42)
    # base_model is simple DDP
    with torch.device("cuda"):
        base_model = nn.Sequential(
            MLP(mlp_dim),
            nn.Sequential(MLP(mlp_dim), nn.Linear(mlp_dim, mlp_dim)),
            MLP(mlp_dim),
        )
        base_optim = torch.optim.Adam(
            base_model.parameters(), weight_decay=0.01, lr=0.01
        )

    fsdp_model_to_save = copy.deepcopy(base_model)
    dist_breakpoint()
    for module in fsdp_model_to_save:
        fully_shard(module)
    fully_shard(fsdp_model_to_save)
    fsdp_optim_to_save = torch.optim.Adam(
        fsdp_model_to_save.parameters(), weight_decay=0.01, lr=0.01
    )

    # inp is different for each rank
    torch.manual_seed(42 + rank)

    # test get full state dict
    for _ in range(epochs):
        inp = torch.randn((2, mlp_dim), device="cuda")
        base_model(inp).sum().backward()
        for param in base_model.parameters():
            torch.distributed.all_reduce(
                param.grad, op=torch.distributed.ReduceOp.AVG
            )
        base_optim.step()
        base_optim.zero_grad()
        fsdp_model_to_save(inp).sum().backward()
        fsdp_optim_to_save.step()
        fsdp_optim_to_save.zero_grad()
    expected_model_sd = base_model.state_dict()
    expected_optim_sd = base_optim.state_dict()
    model_full_sd = utils.get_full_model_state_dict(
        fsdp_model_to_save, is_rank_zero
    )
    optim_full_sd = utils.get_full_optimizer_state_dict(
        fsdp_optim_to_save,
        is_rank_zero,
    )
    if is_rank_zero:
        assert set(model_full_sd.keys()) == set(expected_model_sd.keys())
        for key, value in model_full_sd.items():
            assert value == expected_model_sd[key]
        assert len(optim_full_sd["param_groups"]) == 1
        assert len(optim_full_sd["param_groups"]) == len(expected_optim_sd["param_groups"])
        assert len(optim_full_sd["param_groups"][0].keys()) == len(expected_optim_sd["param_groups"][0].keys())
        for key, value in optim_full_sd["param_groups"][0].items():
            if key == "params":
                assert len(value) == len(expected_optim_sd["param_groups"][0][key])
            else:
                assert value == expected_optim_sd["param_groups"][0][key]
        assert len(optim_full_sd["state"].keys()) == len(expected_optim_sd["state"].keys())
        for actual, expected in zip(
            optim_full_sd["state"].values(), expected_optim_sd["state"].values()
        ):
            assert actual == expected
    else:
        assert len(model_full_sd) == 0
        assert len(optim_full_sd) == 0

    # test set full state dict
    with torch.device("meta"):
        fsdp_model_to_load = nn.Sequential(
            MLP(mlp_dim),
            nn.Sequential(MLP(mlp_dim), nn.Linear(mlp_dim, mlp_dim)),
            MLP(mlp_dim),
        )
    for module in fsdp_model_to_load:
        fully_shard(module)
    fully_shard(fsdp_model_to_load)
    utils.load_from_full_model_state_dict(
        fsdp_model_to_load,
        copy.deepcopy(base_model.state_dict()),
        torch.device("cuda"),
        is_rank_zero,
    )
    fsdp_optim_to_load = torch.optim.Adam(
        fsdp_model_to_load.parameters(), weight_decay=0.01, lr=0.01
    )
    utils.load_from_full_optimizer_state_dict(
        fsdp_optim_to_load,
        # mimic mmap=True where every rank see full SD
        copy.deepcopy(broadcast_full_state_dict(optim_full_sd)),
        torch.device("cuda"),
    )
    for _ in range(epochs):
        inp = torch.randn((2, mlp_dim), device="cuda")
        fsdp_model_to_load(inp).sum().backward()
        fsdp_model_to_save(inp).sum().backward()
        fsdp_optim_to_load.step()
        fsdp_optim_to_save.step()
        fsdp_optim_to_load.zero_grad()
        fsdp_optim_to_save.zero_grad()
    sharded_optim_sd = fsdp_optim_to_load.state_dict()
    expected_sharded_optim_sd = fsdp_optim_to_save.state_dict()
    assert sharded_optim_sd["param_groups"] == expected_sharded_optim_sd["param_groups"]
    assert set(sharded_optim_sd["state"].keys()) == set(expected_sharded_optim_sd["state"].keys())
    for key, value in sharded_optim_sd["state"].items():
        assert value == expected_sharded_optim_sd["state"][key]

    sharded_model_sd = fsdp_model_to_load.state_dict()
    expected_sharded_model_sd = fsdp_model_to_save.state_dict()
    assert set(sharded_model_sd.keys()) == set(expected_sharded_model_sd.keys())
    for key, value in sharded_model_sd.items():
        assert value == expected_sharded_model_sd[key]

if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))
    torch.cuda.set_device(torch.distributed.get_rank())
    dist_print(f"Running test_lora_state_dict on rank {torch.distributed.get_rank()} {torch.distributed.get_world_size()}")
    test_lora_state_dict()
