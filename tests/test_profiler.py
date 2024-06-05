from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

import torchtune
from torchtune import config

FIXTURES_DIR = Path(__file__).parent / "assets"


@pytest.fixture
def valid_profiler_config():
    return OmegaConf.load(FIXTURES_DIR / "valid_profiler.yaml")


@pytest.fixture
def invalid_profiler_config():
    return OmegaConf.load(FIXTURES_DIR / "invalid_profiler.yaml")


TEST_PROFILE_CFG = """
profile:
  enabled: True
  CPU: True
  CUDA: True
  #output_dir: ${artifact_dir}/profiling
  #torch.profiler.profile
  profiler:
    _component_: torch.profiler.profile
    profile_memory: False
    with_stack: True
    record_shapes: False
    with_flops: True
  #torch.profiler.schedule
  schedule:
    _component_: torch.profiler.schedule
    wait: 3
    warmup: 1
    active: 1
    repeat: 0
"""


def test_valid_profile():
    cfg = OmegaConf.create(TEST_PROFILE_CFG)

    torch_profiler_cfg = cfg.profile.profiler
    schedule_cfg = cfg.profile.schedule

    ref_schedule = torch.profiler.schedule(wait=3, warmup=1, active=1, repeat=0)
    test_schedule = config.instantiate(schedule_cfg)
    ref_actions = [ref_schedule(i) for i in range(10)]
    test_actions = [test_schedule(i) for i in range(10)]
    assert ref_actions == test_actions

    ref_profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=ref_schedule,
        profile_memory=False,
        with_stack=True,
        record_shapes=False,
        with_flops=True,
    )
    test_activities = []
    if cfg.profile.CPU:
        test_activities.append(torch.profiler.ProfilerActivity.CPU)
    if cfg.profile.CUDA:
        test_activities.append(torch.profiler.ProfilerActivity.CUDA)
    test_profiler = config.instantiate(
        torch_profiler_cfg, activities=test_activities, schedule=test_schedule
    )

    for _attr in [
        "activities",
        "profile_memory",
        "with_stack",
        "record_shapes",
        "with_flops",
    ]:
        assert getattr(ref_profiler, _attr) == getattr(test_profiler, _attr)


# # def test_invalid_profile(invalid_profile_config):
# #     print(invalid_profile_config)
