## [FEATURE] Enable more configurable profiler

Add more configuration options for `torch.profiler.profile` in `config` files.

Currently `torchtune.utils.profiler` has hardcoded options.  

This PR adds the ability to fully configure the profiler by expanding the accepted args under the `profiler` key in `config` files.

Features:
- `torch.profiler.profile` settings such as `with_stack`, `profile_memory`, etc. can be specified under `profiler.profile`
- Custom profiling [schedule](https://pytorch.org/docs/stable/profiler.html#torch.profiler.schedule) under `profiler.schedule`
- Sensible defaults: the profiler is instantiated with sensible defaults such that user does not need to specify all sections
  - E.g., if only `profiler.enabled=true` but no other keys are included, will default to the current `torchtune.utils.profiler` 
- Works in both single device and distributed settings 
- API is identical to that of current `torchtune` profiler such that the only change required is to use the newly introduced `setup_torch_profiler` to instantiate the profiler within the `recipe`.
  
## Usage

Changes needed to profile a `recipe`:
- Add the a `profiler` section to `config` yaml (see below)
- Call `torchtune.utils._profiler.setup_torch_profiler` within the `setup` method of the `recipe` with theh top-level `args`
- Wrap the training loop with the profiler and `step` the profiler after each `optimizer` step (per the current profiler `recipe`).

### Profiler Config
```
profiler:
    enabled: bool

    #Output directory of trace artifacts
    output_dir: str

    #`torch.profiler.ProfilerActivity` types to trace
    CPU: bool
    CUDA: bool

    #`torch.profiler.profile` options
    profile:
        # _component_ is optional as the setup method will handle
        _component_: torch.profiler.profile
        profile_memory: bool
        with_stack: bool
        record_shapes: bool
        with_flops: bool
    
    #`torch.profiler.schedule` options
    schedule:
        # _component_ is optional as the setup method will handle
        _component_: torch.profiler.schedule
        wait: int
        warmup: int
        active: int
        repeat: int
```
**Notes**
- All args are optional and if not specified, will default to minimal working set per `torch.profiler.profile` defaults (see the added `_DEFAULTS` in `torchtune.utils._profiler`)
- Certain options are overridden to match requirements of the torch profiler
  - E.g., `profile_memory` requires `with_stack` and `record_shapes` so if user sets `profile_memory: true`, then the latter two will automatically be set within the setup method
- Top-level `cfg` will be modified in-place by `setup_torch_profiler` to reflect updated `profiler` config for logging / debugging purposes

## Examples
- Run single device, same as current `torchtune.utils.profiler`:
    ```
    tune run lora_finetune_single_device \
    --config llama2/7B_lora_single_device \
    profiler.enabled=True
    ```
- Same as above but with profiler disabled:
    ```
    tune run lora_finetune_single_device \
    --config llama2/7B_lora_single_device \
    profiler.enabled=False
    ```

- Run single device, profiling on a 10-step cycle:  
    ```
    tune run lora_finetune_single_device \
    --config llama2/7B_lora_single_device \
    profiler.enabled=True \
    profiler.output_dir=./profiler_outputs \
    profiler.schedule.wait=5 \
    profiler.schedule.warmup=4 \
    profiler.schedule.active=1 \
    profiler.schedule.repeat=0
    ```
    At the end of each cycle, a `tensorboard` / `chrome` trace along with key averages table of operators ordered by `cuda` time will be dumped to `profiler_outputs`.

- Run distributed, profiling for single cycle:
    ```
    tune run --nnodes=1 --nproc_per_node=2 lora_finetune_fsdp2 \
    --config llama2/7B_lora \
    profiler.enabled=True \
    profiler.profile_memory=True \
    profiler.schedule.wait=10 \
    profiler.schedule.warmup=10 \
    profiler.schedule.active=1 \
    profiler.schedule.repeat=1
    ```
    In addition to the previous outputs, additional memory usage info will be recorded and a viewable [memory timeline](https://pytorch.org/docs/stable/profiler.html#torch.profiler._KinetoProfile.export_memory_timeline) will be exported as well.

    Each trace will be marked with the `rank` of the process.
    
    Note that profiling memory adds significant overhead (training time and large trace files) so the profiling schedule should be set accordingly. 

## Tests

See `tests/test_profiler.py` for unit testing of profiler instantiation with `torchtune.config` as well as the expected functionality of `torchtune.utils.setup_torch_profiler`.

## Next Steps

This PR edits `recipes/lora_finetune_single_device.py` and `recipes/dev/lora_finetune_fsdp2.py` to demonstrate usage.  If there is interest, can add profiling functionality to other registered recipes as well.