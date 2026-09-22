"""TextCLF TQ vLLM quantization plugin.

The Python integration is intentionally public. Proprietary TQ execution and
authorization live in the compiled ``tq_kernels`` extension. GGUF is not
supported.
"""

from .tq import TqConfig, TqLinearMethod, configure_tq_pipeline_partition
from . import runtime_ops as _runtime_ops  # registers FakeTensor metadata


def register() -> None:
    """Register TQ with vLLM before EngineCore workers are spawned."""
    try:
        configure_tq_pipeline_partition()
    except Exception as exc:
        # Auto partitioning is an optimization; configuration gets another
        # chance during TqConfig initialization if early inspection is unavailable.
        print("[TQ PP] early auto-partition skipped:", repr(exc), flush=True)

    from vllm import ModelRegistry

    if "tq" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "tq",
            "vllm_add_dummy_model.tq:tq",
        )


__all__ = ["TqConfig", "TqLinearMethod", "configure_tq_pipeline_partition", "register"]
