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

_DEFAULT_PROFILER_ACTIVITIES = [
    torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,
]

_DEFAULT_SCHEDULE: dict = {
    "_component_": "torch.profiler.schedule",
    "wait": 10,
    "warmup": 5,
    "active": 3,
    "repeat": 1,
}
_DEFAULT_SCHEDULE_CFG = DictConfig(_DEFAULT_SCHEDULE)
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
    _, rank = utils.get_world_size_and_rank()
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

    # Construct the memory timeline file.
    if prof.profile_memory:
        try:
            prof.export_memory_timeline(
                f"{curr_trace_dir}/rank{rank}_memory-timeline.html"
            )
        except:
            _warn("Failed to export memory timeline to html, retrying as gzipped json.")
            try:
                prof.export_memory_timeline(
                    f"{curr_trace_dir}/rank{rank}_memory-timeline.json.gz"
                )
            except:
                _warn(
                    "Failed to export memory timeline to gzipped json. Saving profiler timeline object instead."
                )
                from torch.profiler._memory_profiler import MemoryProfileTimeline

                memory_profile = MemoryProfileTimeline(prof._memory_profile())
                torch.save(
                    memory_profile, f"{curr_trace_dir}/rank{rank}_memory-timeline.pt"
                )

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
    torch.distributed.barrier()


def setup_torch_profiler(cfg: DictConfig) -> torch.profiler.profile:
    """
    Sets up torch.profiler.profile

    Args:
        cfg (DictConfig): profiler config with expected structure:
            ```
            profile:
                output_dir: str
                CPU: bool
                CUDA: bool

                profiler:
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
        Note:
            - the profiler schedule updates with respect to an optimizer step:
                - e.g., if `gradient_accumulation = 2`, then the profiler will step every 2 batches.
            - sensible defaults will be chosen if the config is missing options
                - if no activities are specified, profiler will default to CPU + CUDA
                - if no schedule is specified, profiler will default to wait 10, warmup 5, active 3, repeat 1
                - if a schedule is specified, profiler will validate that the schedule is valid and can be passed to `instantiate`
                - certain options will be overridden (`with_stack` and `record_shapes`) depending on requirements of other options
                    - e.g., `profile_memory` requires `with_stack` and `record_shapes`

    Returns:
        torch.profiler.profile
    """
    torch_profiler_cfg = OmegaConf.select(
        cfg, "profiler", default=None, throw_on_missing=False
    )
    assert (
        torch_profiler_cfg is not None
    ), "Missing torch profiler config, please make sure to include a valid profiler config under the 'profile.profiler' key"

    # Set up profiler activities
    activities = []
    if cfg.CPU:
        activities.append(torch.profiler.ProfilerActivity.CPU)
    if cfg.CUDA:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    if len(activities) == 0:
        _warn("No activities specified, defaulting to CPU + CUDA")
        activities = _DEFAULT_PROFILER_ACTIVITIES

    # Set up profiler schedule
    schedule_cfg = OmegaConf.select(
        cfg, "schedule", default=None, throw_on_missing=False
    )

    # Use default schedule if None, else validate that schedule is valid and can be passed to `instantiate`
    if schedule_cfg is None:
        _warn(
            f" No schedule found in profiler config, loading default schedule {_DEFAULT_SCHEDULE_CFG}"
        )
        schedule_cfg = _DEFAULT_SCHEDULE_CFG
    else:
        assert all(
            k in schedule_cfg for k in ["wait", "warmup", "active"]
        ), "Invalid schedule config: must specify wait, warmup, active"
        if "repeat" not in schedule_cfg:
            _warn(
                " No repeat found in schedule config, setting to 1."
            )
            schedule_cfg["repeat"] = 1

    schedule = config.instantiate(schedule_cfg) if schedule_cfg is not None else None

    profile_memory = OmegaConf.select(
        torch_profiler_cfg, "profile_memory", default=False
    )

    # profile_memory requires with_stack and record_shapes, hence we override these if profile_memory is True
    # See torch.profiler.profiler._memory_profile
    with_stack = (
        OmegaConf.select(torch_profiler_cfg, "with_stack", default=False)
        or profile_memory
    )
    record_shapes = (
        OmegaConf.select(torch_profiler_cfg, "record_shapes", default=False)
        or profile_memory
    )

    # experimental config is needed to export stacks: see https://github.com/pytorch/pytorch/issues/100253
    experimental_config = _ExperimentalConfig(verbose=True) if with_stack else None

    # Handle exporting of trace, memory timeline and other profiler artifacts
    profiler_output_dir = OmegaConf.select(
        cfg, "output_dir", default=None, throw_on_missing=False
    )
    if profiler_output_dir is None:
        _warn(
            f" No output directory found in profiler config, defaulting to {_DEFAULT_PROFILE_DIR}"
        )
        profiler_output_dir = _DEFAULT_PROFILE_DIR

    output_dir = Path(profiler_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    callback = partial(trace_handler, output_dir=output_dir)

    # Update profiler cfg
    cfg.output_dir = profiler_output_dir
    cfg.schedule = schedule_cfg
    cfg.profiler.profile_memory = profile_memory
    cfg.profiler.with_stack = with_stack
    cfg.profiler.record_shapes = record_shapes

    profiler = config.instantiate(
        torch_profiler_cfg,
        activities=activities,
        schedule=schedule,
        profile_memory=profile_memory,
        with_stack=with_stack,
        record_shapes=record_shapes,
        experimental_config=experimental_config,
        on_trace_ready=callback,
    )

    return profiler
