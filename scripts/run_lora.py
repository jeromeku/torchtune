import argparse
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

TUNE_PATH = "torchtune/_cli/tune.py"


def main(args):
    overrides = args.overrides

    cmd = f"tune run --nnodes {args.nnodes} --nproc_per_node {args.ngpus} {args.recipe} --config {args.config}"
    if overrides is not None:
        cmd += " " + " ".join(overrides)

    print(f"{cmd.strip()}")
    
    with patch.object(sys, "argv", cmd.split()):
        runpy.run_path(TUNE_PATH, run_name="__main__")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--recipe", type=str, default="my_recipes/lora_finetune_distributed.py")
    parser.add_argument("--config", type=str, default="my_configs/qwen3_1.7B_lora.yaml")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    main(args)
    