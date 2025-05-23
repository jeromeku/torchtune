from viztracer import VizTracer
from viztracer import get_tracer as get_global_tracer


def _get_trace_config(
    include_files=None,
    ignore_frozen=True,
    ignore_c_function=True,
    log_func_args=True,
    log_torch=False,
    log_func_retval=True,
    dump_raw=False,
    output_file=None,
    **kwargs,
):
    tracer_kwargs = dict(
        include_files=include_files,
        ignore_frozen=ignore_frozen,
        ignore_c_function=ignore_c_function,
        log_func_args=log_func_args,
        log_func_retval=log_func_retval,
        log_torch=log_torch,
        dump_raw=dump_raw,
        output_file=output_file,
        **kwargs,
    )
    return tracer_kwargs


def get_tracer(
    include_files=None,
    ignore_frozen=True,
    ignore_c_function=True,
    log_func_args=True,
    log_torch=False,
    log_func_retval=True,
    dump_raw=False,
    output_file=None,
    **kwargs,
):
    tracer = VizTracer(
        include_files=include_files,
        ignore_frozen=ignore_frozen,
        ignore_c_function=ignore_c_function,
        log_func_args=log_func_args,
        log_func_retval=log_func_retval,
        log_torch=log_torch,
        dump_raw=dump_raw,
        output_file=output_file,
        **kwargs
    )

def configure_tracer(tracer: VizTracer = None, **kwargs):
    if tracer is None:
        tracer = get_global_tracer()
    
    for k,v in kwargs.items():
        setattr(tracer, k, v)

    return tracer