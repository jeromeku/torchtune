import pytest
import torch
from omegaconf import OmegaConf
from torch._C._profiler import _ExperimentalConfig

from torchtune import config
from torchtune.utils.profiling_utils import setup_torch_profiler

## Test fixtures

PROFILER_ATTRS = [
    "activities",
    "profile_memory",
    "with_stack",
    "record_shapes",
    "with_flops",
]


@pytest.fixture
def profiler_cfg():
    return """
profile:
  enabled: True
  CPU: True
  CUDA: True
  profiler:
    _component_: torch.profiler.profile
    profile_memory: False
    with_stack: False
    record_shapes: True
    with_flops: True
  schedule:
    _component_: torch.profiler.schedule
    wait: 3
    warmup: 1
    active: 1
    repeat: 0
"""


@pytest.fixture
def reference_profiler_basic():
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=3, warmup=1, active=1, repeat=0),
        profile_memory=False,
        with_stack=False,
        record_shapes=True,
        with_flops=True,
    )


@pytest.fixture
def reference_profiler_full():
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=3, warmup=1, active=1, repeat=0),
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
        with_flops=True,
        experimental_config=_ExperimentalConfig(verbose=True),
    )


def check_profiler_attrs(profiler, ref_profiler):
    for attr in PROFILER_ATTRS:
        assert getattr(profiler, attr) == getattr(ref_profiler, attr)


def check_schedule(schedule, ref_schedule, num_steps=10):
    ref_steps = [ref_schedule(i) for i in range(num_steps)]
    test_steps = [schedule(i) for i in range(num_steps)]
    assert ref_steps == test_steps


## Tests
def test_instantiate(profiler_cfg, reference_profiler_basic):
    cfg = OmegaConf.create(profiler_cfg)

    torch_profiler_cfg = cfg.profile.profiler
    schedule_cfg = cfg.profile.schedule

    ref_schedule = torch.profiler.schedule(wait=3, warmup=1, active=1, repeat=0)
    test_schedule = config.instantiate(schedule_cfg)
    check_schedule(ref_schedule, test_schedule)

    test_activities = []
    if cfg.profile.CPU:
        test_activities.append(torch.profiler.ProfilerActivity.CPU)
    if cfg.profile.CUDA:
        test_activities.append(torch.profiler.ProfilerActivity.CUDA)
    test_profiler = config.instantiate(
        torch_profiler_cfg, activities=test_activities, schedule=test_schedule
    )
    check_profiler_attrs(test_profiler, reference_profiler_basic)


def test_instantiate_full(profiler_cfg, reference_profiler_full):
    cfg = OmegaConf.create(profiler_cfg)

    # Check `setup` automatically overrides `with_stack` and `record_shapes` when profile_memory is True and adds
    # experimental_config, which is needed for stack exporting (see comments in `setup_torch_profiler`)
    cfg.profile.profiler.profile_memory = True
    cfg.profile.profiler.with_stack = False
    cfg.profile.profiler.record_shapes = False
    profiler = setup_torch_profiler(cfg.profile)

    check_profiler_attrs(profiler, reference_profiler_full)
    assert profiler.experimental_config is not None


def test_schedule_setup(profiler_cfg, reference_profiler_basic):
    cfg = OmegaConf.create(profiler_cfg)

    profiler = setup_torch_profiler(cfg.profile)

    check_profiler_attrs(profiler, reference_profiler_basic)

    # Test that after removing schedule, setup method will implement default schedule
    from torchtune.utils.profiling_utils import _DEFAULT_SCHEDULE_CFG

    cfg.profile.pop("schedule")
    profiler = setup_torch_profiler(cfg.profile)
    assert cfg.profile.schedule == _DEFAULT_SCHEDULE_CFG

    default_schedule = config.instantiate(_DEFAULT_SCHEDULE_CFG)
    check_schedule(profiler.schedule, default_schedule)

    # Test invalid schedule
    cfg.profile.schedule.pop("wait")
    with pytest.raises(ValueError):
        profiler = setup_torch_profiler(cfg.profile)

    # Test missing `repeat` replaced with `1`
    cfg.profile.schedule = _DEFAULT_SCHEDULE_CFG
    cfg.profile.schedule.pop("repeat")
    profiler = setup_torch_profiler(cfg.profile)
    assert cfg.profile.schedule.repeat == 1


def test_defaults_setup(profiler_cfg):
    cfg = OmegaConf.create(profiler_cfg)

    from torchtune.utils.profiling_utils import (
        _DEFAULT_PROFILE_DIR,
        _DEFAULT_PROFILER_ACTIVITIES,
        _DEFAULT_PROFILER_OPTS,
    )

    # Test setup automatically adds CPU + CUDA tracing if neither CPU nor CUDA is specified
    cfg.profile.pop("CPU")
    cfg.profile.pop("CUDA")
    profiler = setup_torch_profiler(cfg.profile)
    assert profiler.activities == _DEFAULT_PROFILER_ACTIVITIES

    # Test cfg output_dir is set correctly
    if not OmegaConf.is_missing(cfg, "profile.output_dir"):
        cfg.profile.pop("output_dir")
    profiler = setup_torch_profiler(cfg.profile)
    assert cfg.profile.output_dir == _DEFAULT_PROFILE_DIR

    # Test missing profiler options are set to defaults
    torch_profiler_cfg = cfg.profile.profiler
    torch_profiler_cfg.pop("profile_memory")
    torch_profiler_cfg.pop("with_stack")
    torch_profiler_cfg.pop("record_shapes")
    torch_profiler_cfg.pop("with_flops")
    profiler = setup_torch_profiler(cfg.profile)
    assert torch_profiler_cfg.profile_memory == _DEFAULT_PROFILER_OPTS["profile_memory"]
    assert torch_profiler_cfg.with_stack == _DEFAULT_PROFILER_OPTS["with_stack"]
    assert torch_profiler_cfg.record_shapes == _DEFAULT_PROFILER_OPTS["record_shapes"]
    assert torch_profiler_cfg.with_flops == _DEFAULT_PROFILER_OPTS["with_flops"]


def test_fake_profiler():
    pass
