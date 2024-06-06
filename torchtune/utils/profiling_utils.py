# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import tensorboard_trace_handler

from torchtune import config, utils

log = utils.get_logger("INFO")

PROFILER_KEY = "profiler"
_DEFAULT_PROFILER_ACTIVITIES = {
    torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,
}

_DEFAULT_SCHEDULE_SINGLE: dict = {
    # "_component_": "torch.profiler.schedule",
    "wait": 100,
    "warmup": 5,
    "active": 5,
    "repeat": 1,
}

_DEFAULT_SCHEDULE_DISTRIBUTED: dict = {
    # "_component_": "torch.profiler.schedule",
    "wait": 5,
    "warmup": 5,
    "active": 1,
    "repeat": 1,
}
_DEFAULT_PROFILER_OPTS: dict = {
    "profile_memory": False,
    "with_stack": False,
    "record_shapes": True,
    "with_flops": False,
}
_DEFAULT_SCHEDULE_CFG_SINGLE = DictConfig(_DEFAULT_SCHEDULE_SINGLE)
_DEFAULT_SCHEDULE_CFG_DISTRIBUTED = DictConfig(_DEFAULT_SCHEDULE_DISTRIBUTED)

_DEFAULT_PROFILE_DIR: str = "profiler_output"


def _warn(msg: str):
    _, rank = utils.get_world_size_and_rank()
    if rank == 0:
        log.warn(msg)


def trace_handler(
    prof: torch.profiler.profiler.profile,
    output_dir,
    metric="self_cuda_time_total",
    row_limit=-1,
):
    world_size, rank = utils.get_world_size_and_rank()
    curr_trace_dir_name = "iteration_" + str(prof.step_num)
    curr_trace_dir = os.path.join(output_dir, curr_trace_dir_name)
    if not os.path.exists(curr_trace_dir):
        os.makedirs(curr_trace_dir, exist_ok=True)

    # Export chrome / tensorboard trace
    if rank == 0:
        log.info(f"Dumping traces at step {prof.step_num}")
    begin = time.monotonic()

    # Use tensorboard trace handler rather than directly exporting chrome traces since
    # tensorboard doesn't seem to be able to parse traces with prof.export_chrome_trace
    exporter = tensorboard_trace_handler(
        curr_trace_dir, worker_name=f"rank{rank}", use_gzip=True
    )
    exporter(prof)

    if rank == 0:
        log.info(f"Finished dumping traces in {time.monotonic() - begin:.2f} seconds")

    # Memory timeline sometimes fails to export
    if prof.profile_memory:
        if rank == 0:
            try:
                prof.export_memory_timeline(
                    f"{curr_trace_dir}/rank{rank}_memory-timeline.html"
                )
            except Exception as e:
                log.warn(f" Failed to export memory timeline: {e}")

    # Dump stack traces
    if prof.with_stack:
        prof.export_stacks(f"{curr_trace_dir}/rank{rank}_stacks.txt", metric=metric)

    # Export event averages
    key_avgs = prof.key_averages(
        group_by_input_shape=prof.record_shapes, group_by_stack_n=5
    ).table(sort_by=metric, row_limit=row_limit)
    with open(f"{curr_trace_dir}/rank{rank}_key_averages.txt", "w") as f:
        print(key_avgs, file=f)
    if rank == 0:
        log.info(f"Saving profiling results to {curr_trace_dir}")

    # TODO: Is this necessary?
    # see https://github.com/pytorch/torchtitan/blob/3050098dcee4901d88c712f9e8e9703d1735a29b/torchtitan/profiling.py#L48
    if world_size > 1:
        torch.distributed.barrier()


class FakeProfiler:
    """
    Mock object that minimally mimics behavior of torch.profiler.profile

    Essentially a contextlib.nullcontext object with a `step` method
    """

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

    def step(self):
        pass


def should_profile(cfg: DictConfig) -> bool:
    return cfg.get(PROFILER_KEY, None) is not None and cfg[PROFILER_KEY].get(
        "enabled", True
    )


def setup_torch_profiler(cfg: DictConfig) -> torch.profiler.profile:
    """
    Sets up torch.profiler.profile

    NOTE: Enabling the profiler may have training speed reduction.

    Args:
        cfg (DictConfig): profiler config with following options:
            ```
            profiler:
                output_dir: str
                CPU: bool
                CUDA: bool

                profile:
                    _component_: torch.profiler.profile
                    profile_memory: bool
                    with_stack: bool
                    record_shapes: bool
                    with_flops: bool
                schedule:
                    _component_: torch.profiler.schedule
                    wait: int
                    warmup: int
                    active: int
                    repeat: int
            ```
    Returns:
        torch.profiler.profile | FakeProfiler
    Notes:
        - `cfg` is modified in-place with the defaults per the comments below
        - the profiler schedule updates with respect to an optimizer step:
            - e.g., if `gradient_accumulation = 2`, then the profiler will step every 2 batches.
        - sensible defaults will be chosen if the config is missing options
            - if no activities are specified, profiler will default to CPU + CUDA
            - if no schedule is specified, profiler will default to wait 10, warmup 5, active 3, repeat 1
            - if a schedule is specified, profiler will validate that the schedule is valid and can be passed to `instantiate`
            - certain options will be overridden (`with_stack` and `record_shapes`) depending on requirements of other options
                - e.g., `profile_memory` requires `with_stack` and `record_shapes`
        - if no profiler config is found or the `cfg.enabled=False`, a fake profiler will be returned that
        minimally mimicks the interface of torch.profiler.profile (context decorator with `step` method)
    """

    if not should_profile(cfg):
        return FakeProfiler()

    cfg[PROFILER_KEY].enabled = cfg[PROFILER_KEY].get("enabled", True)
    torch_profiler_cfg = cfg[PROFILER_KEY].get("profile", None)
    if torch_profiler_cfg is None:
        _warn(
            f" Missing torch profiler config, instantiating with default settings: {_DEFAULT_PROFILER_OPTS}"
        )
        cfg[PROFILER_KEY].profile = torch_profiler_cfg = OmegaConf.create(
            _DEFAULT_PROFILER_OPTS
        )

    # Set up profiler activities
    activities = []
    profile_cpu = cfg[PROFILER_KEY].get("CPU", False)
    profile_cuda = cfg[PROFILER_KEY].get("CUDA", False)
    if profile_cpu:
        activities.append(torch.profiler.ProfilerActivity.CPU)
    if profile_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    if len(activities) == 0:
        _warn("No activities specified, defaulting to CPU + CUDA")
        activities = _DEFAULT_PROFILER_ACTIVITIES

    # Set up profiler schedule
    schedule_cfg = cfg[PROFILER_KEY].get("schedule", None)

    # Use default schedule if None, else validate that schedule is valid and can be passed to `instantiate`
    if schedule_cfg is None:
        world_size, _ = utils.get_world_size_and_rank()
        if world_size > 1:
            _DEFAULT_SCHEDULE_CFG = _DEFAULT_SCHEDULE_CFG_DISTRIBUTED
        else:
            _DEFAULT_SCHEDULE_CFG = _DEFAULT_SCHEDULE_CFG_SINGLE
        _warn(
            f" No schedule found in profiler config, loading default schedule {_DEFAULT_SCHEDULE_CFG}"
        )
        schedule_cfg = _DEFAULT_SCHEDULE_CFG
    else:
        if not all(k in schedule_cfg for k in ["wait", "warmup", "active"]):
            raise ValueError(
                "Invalid schedule config: must specify wait, warmup, active"
            )
        if "repeat" not in schedule_cfg:
            _warn(
                """ No repeat found in schedule config, setting to 1 (one cycle).
                If you want to cycle continuously, specify repeat = 0"""
            )
            schedule_cfg["repeat"] = 1
    if "_component_" not in schedule_cfg:
        schedule_cfg["_component_"] = "torch.profiler.schedule"
    schedule = config.instantiate(schedule_cfg)

    profile_memory = torch_profiler_cfg.get(
        "profile_memory", _DEFAULT_PROFILER_OPTS["profile_memory"]
    )

    # profile_memory requires with_stack and record_shapes, hence we override these if profile_memory is True
    # See torch.profiler.profiler._memory_profile
    with_stack = (
        torch_profiler_cfg.get("with_stack", _DEFAULT_PROFILER_OPTS["with_stack"])
        or profile_memory
    )
    record_shapes = (
        torch_profiler_cfg.get("record_shapes", _DEFAULT_PROFILER_OPTS["record_shapes"])
        or profile_memory
    )
    with_flops = torch_profiler_cfg.get(
        "with_flops", _DEFAULT_PROFILER_OPTS["with_flops"]
    )

    # experimental config is needed to export stacks: see https://github.com/pytorch/pytorch/issues/100253
    experimental_config = _ExperimentalConfig(verbose=True) if with_stack else None

    # Handle exporting of trace, memory timeline and other profiler artifacts
    profiler_output_dir = cfg[PROFILER_KEY].get("output_dir", None)

    if profiler_output_dir is None:
        _warn(
            f" No output directory found in profiler config, defaulting to {_DEFAULT_PROFILE_DIR}"
        )
        profiler_output_dir = _DEFAULT_PROFILE_DIR

    output_dir = Path(profiler_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    callback = partial(trace_handler, output_dir=output_dir)

    # Update profiler cfg in-place
    cfg[PROFILER_KEY].output_dir = profiler_output_dir
    cfg[PROFILER_KEY].schedule = schedule_cfg
    cfg[PROFILER_KEY].profile.profile_memory = profile_memory
    cfg[PROFILER_KEY].profile.with_stack = with_stack
    cfg[PROFILER_KEY].profile.record_shapes = record_shapes
    cfg[PROFILER_KEY].profile.with_flops = with_flops

    if "_component_" not in torch_profiler_cfg:
        cfg[PROFILER_KEY].profile["_component_"] = "torch.profiler.profile"

    profiler = config.instantiate(
        cfg[PROFILER_KEY].profile,
        activities=activities,
        schedule=schedule,
        experimental_config=experimental_config,
        on_trace_ready=callback,
    )

    return profiler
