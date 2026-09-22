from dataclasses import dataclass
import json
import os

# Self-identifying Python-only runtime build marker. This intentionally prints
# once per importing process so deployment logs prove which tq.py is active.
_TQ_RUNTIME_BUILD = "qwen38-ngram-v9-dynamic-m"
print(f"[TQ RUNTIME BUILD] {_TQ_RUNTIME_BUILD}", flush=True)
from pathlib import Path
import re
import threading
import time
import types
import concurrent.futures
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from huggingface_hub import snapshot_download
import multiprocessing


def _tq_is_ple_offload_worker() -> bool:
    return multiprocessing.current_process().name == "PleOffloadWorker"


def _tq_runtime_device() -> torch.device:
    """Return the process-local device for TQ parameter construction.

    The PLE offload process constructs a model replica for loading/offload
    bookkeeping and must not allocate TQ-owned tensors on CUDA. Normal vLLM
    workers/EngineCore keep the original CUDA behavior.
    """
    return torch.device("cpu" if _tq_is_ple_offload_worker() else "cuda")


def _tq_log(*args) -> None:
    proc = multiprocessing.current_process().name
    ts = time.strftime("%H:%M:%S")
    print(f"[TQ][{ts}][{proc}]", *args, flush=True)


# Hugging Face Hub calls issued during process_weights_after_loading (one
# per quantized layer/expert shard) have no built-in bound on how long they
# can take: a network stall, DNS hiccup, rate limit, or a stale filelock
# left behind by a previously interrupted download can all block the
# underlying call forever. Because these calls happen inside vLLM's
# weight-loading path -- with no surrounding timeout of its own -- a single
# stuck call here silently freezes the whole worker process at 0% GPU
# utilization with no further log output. That is indistinguishable from a
# genuine deadlock unless the call itself is bounded and logged.
_TQ_HF_DOWNLOAD_TIMEOUT = float(os.environ.get("TQ_HF_DOWNLOAD_TIMEOUT", "300"))


def _tq_run_with_timeout(fn, *, desc: str, timeout: float = _TQ_HF_DOWNLOAD_TIMEOUT):
    """Run a blocking call with a hard wall-clock timeout.

    Python cannot forcibly kill a running thread, so on timeout the
    underlying call may still be running in the background -- but the
    caller gets a clear, actionable TimeoutError immediately instead of an
    indefinite, unexplained hang with no diagnostic trail.
    """
    start = time.monotonic()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn)
    try:
        result = future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        elapsed = time.monotonic() - start
        pool.shutdown(wait=False, cancel_futures=False)
        raise TimeoutError(
            f"TQ: {desc} did not complete within {timeout:.0f}s "
            f"(elapsed={elapsed:.0f}s). This usually means a network stall "
            f"talking to the Hugging Face Hub, or a stale lock file left "
            f"behind under the HF cache directory by a previously "
            f"interrupted download. Set TQ_HF_DOWNLOAD_TIMEOUT to raise "
            f"this bound, check network egress from the container, or "
            f"inspect ~/.cache/huggingface/.locks for stale lock files."
        ) from None
    else:
        pool.shutdown(wait=False)
    elapsed = time.monotonic() - start
    _tq_log(f"{desc} completed in {elapsed:.1f}s")
    return result


FIXED_SEED_D_R = 987654321
FIXED_SEED_D_L = 123456789

torch.set_float32_matmul_precision("high")
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)

# ---------------------------------------------------------------------------
# TQ residual-checkpoint guard for native vLLM packed linears
# ---------------------------------------------------------------------------
#
# TQ checkpoints intentionally keep zero-sized residual ``.weight`` tensors for
# matrices whose real representation lives in quantization_data.  Most TQ-owned
# linears replace the runtime parameter's loader with ``_ignore_hf_weight_loader``.
# Qwen3.8-Flash-Next, however, uses nested AutoWeightsLoader mappings and native
# merged-column modules; on that path a zero-sized checkpoint tensor can reach
# vLLM's stock ``weight_loader_v2`` before the per-parameter TQ sink is consulted.
# The stock loader then calls ``loaded_weight.narrow(...)`` and fails with:
#
#   RuntimeError: start (0) + length (...) exceeds dimension size (0)
#
# Keep the compatibility guard extremely narrow:
#   * it only runs while a TQ quantization config is active in this process,
#   * it only patches MergedColumnParallelLinear.weight_loader_v2, and
#   * it only consumes incoming tensors with numel() == 0.
#
# Non-empty tensors always delegate to the original vLLM loader unchanged.
_TQ_RUNTIME_ACTIVE = False
_TQ_ORIGINAL_MERGED_WEIGHT_LOADER_V2 = None


def _tq_set_runtime_active(active: bool = True) -> None:
    global _TQ_RUNTIME_ACTIVE
    _TQ_RUNTIME_ACTIVE = bool(active)


def _install_tq_zero_checkpoint_weight_guard() -> None:
    """Guard native merged-column loaders against TQ zero placeholders.

    This is a compatibility shim for vLLM branches whose Qwen3.8 nested
    AutoWeightsLoader can route a TQ residual placeholder through the stock
    merged-column loader.  It is intentionally a no-op for non-TQ runs and for
    every non-empty checkpoint tensor.
    """
    global _TQ_ORIGINAL_MERGED_WEIGHT_LOADER_V2

    try:
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear
    except Exception:
        return

    current = getattr(MergedColumnParallelLinear, "weight_loader_v2", None)
    if current is None:
        return

    if getattr(current, "_tq_zero_checkpoint_guard", False):
        return

    original = current
    _TQ_ORIGINAL_MERGED_WEIGHT_LOADER_V2 = original

    def _tq_guarded_weight_loader_v2(
        self,
        param,
        loaded_weight,
        *args,
        **kwargs,
    ):
        guard_enabled = os.environ.get(
            "TQ_ZERO_CHECKPOINT_WEIGHT_GUARD", "1"
        ).strip().lower() not in {"0", "false", "off", "no"}

        loaded_shape = tuple(getattr(loaded_weight, "shape", ()))
        try:
            loaded_numel = int(loaded_weight.numel())
        except Exception:
            loaded_numel = None
        zero_sized = (
            any(int(d) == 0 for d in loaded_shape)
            or loaded_numel == 0
        )

        if (
            _TQ_RUNTIME_ACTIVE
            and os.environ.get("TQ_DEBUG_DISPATCH", "0") == "1"
        ):
            shard_id = kwargs.get("shard_id", None)
            if shard_id is None and args:
                shard_id = args[0]
            print(
                "[TQ MERGED LOADER PROBE]",
                "module=", type(self).__name__,
                "param_type=", type(param).__name__,
                "param_shape=", tuple(getattr(param, "shape", ())),
                "loaded_type=", type(loaded_weight).__name__,
                "loaded_shape=", loaded_shape,
                "loaded_numel=", loaded_numel,
                "zero_sized=", zero_sized,
                "shard_id=", shard_id,
                flush=True,
            )

        if guard_enabled and _TQ_RUNTIME_ACTIVE and zero_sized:
            if os.environ.get("TQ_DEBUG_DISPATCH", "0") == "1":
                shard_id = kwargs.get("shard_id", None)
                if shard_id is None and args:
                    shard_id = args[0]
                print(
                    "[TQ ZERO CHECKPOINT WEIGHT SKIP]",
                    "module=", type(self).__name__,
                    "param_shape=", tuple(getattr(param, "shape", ())),
                    "loaded_shape=", loaded_shape,
                    "loaded_numel=", loaded_numel,
                    "shard_id=", shard_id,
                    flush=True,
                )
            # The real matrix is owned by TQ packed parameters. Returning here
            # consumes a residual zero-storage checkpoint placeholder before
            # vLLM attempts merged-column slicing.
            return

        try:
            return original(self, param, loaded_weight, *args, **kwargs)
        except Exception as exc:
            if _TQ_RUNTIME_ACTIVE:
                shard_id = kwargs.get("shard_id", None)
                if shard_id is None and args:
                    shard_id = args[0]
                print(
                    "[TQ MERGED LOADER ERROR]",
                    "module=", type(self).__name__,
                    "param_type=", type(param).__name__,
                    "param_shape=", tuple(getattr(param, "shape", ())),
                    "loaded_type=", type(loaded_weight).__name__,
                    "loaded_shape=", tuple(getattr(loaded_weight, "shape", ())),
                    "shard_id=", shard_id,
                    "error=", repr(exc),
                    flush=True,
                )
            raise

    _tq_guarded_weight_loader_v2._tq_zero_checkpoint_guard = True
    _tq_guarded_weight_loader_v2._tq_original = original
    MergedColumnParallelLinear.weight_loader_v2 = _tq_guarded_weight_loader_v2


_install_tq_zero_checkpoint_weight_guard()

from vllm.model_executor.layers.quantization import (
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)

try:
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
except Exception:
    VocabParallelEmbedding = ()

# Active TQ config for model-specific constructors that intentionally pass
# quant_config=None (Qwen3.8 PLE does this on the preview branch).
_TQ_ACTIVE_CONFIG = None
_TQ_ORIGINAL_VOCAB_EMBED_INIT = None

def _tq_set_active_config(config) -> None:
    global _TQ_ACTIVE_CONFIG
    _TQ_ACTIVE_CONFIG = config

def _tq_manifest_unique_ngram_group(manifest) -> str | None:
    if manifest is None:
        return None
    groups = set()
    for name in getattr(manifest, "quantized_layers", ()): 
        sname = str(name)
        if not re.search(r"\.shard_\d+$", sname, re.IGNORECASE):
            continue
        try:
            meta = manifest.load_meta(sname)
        except Exception:
            continue
        if meta.get("target_kind") != "tq_ngram_chunked_shard":
            continue
        group = str(meta.get("embedding_group") or re.sub(
            r"\.shard_\d+$", "", sname, flags=re.IGNORECASE
        ))
        groups.add(group)
    if len(groups) == 1:
        return next(iter(groups))
    return None

def _install_tq_ple_vocab_embedding_hook() -> None:
    """Inject TQ into Qwen3.8's huge PLE embedding before allocation.

    The preview Qwen3.8 PLE constructor creates VocabParallelEmbedding with
    quant_config=None, which bypasses TqConfig.get_quant_method() entirely and
    selects UnquantizedEmbeddingMethod. That method immediately allocates the
    ~95 GiB BF16 table. This wrapper only intervenes for the logical n-gram
    embedding (or the distinctive huge PLE table whose embedding dimension
    matches the unique N-gram manifest group), injects the active TqConfig, and supplies the manifest group as prefix when the model omitted
    it. The original vLLM constructor then performs normal quant-method dispatch.
    """
    global _TQ_ORIGINAL_VOCAB_EMBED_INIT
    if not isinstance(VocabParallelEmbedding, type):
        return
    current = getattr(VocabParallelEmbedding, "__init__", None)
    if current is None or getattr(current, "_tq_ple_hook", False):
        return
    original = current
    _TQ_ORIGINAL_VOCAB_EMBED_INIT = original

    def _tq_vocab_embedding_init(self, *args, **kwargs):
        active = _TQ_ACTIVE_CONFIG
        manifest = getattr(active, "manifest", None) if active is not None else None

        def _arg(pos, key, default=None):
            if key in kwargs:
                return kwargs[key]
            if len(args) > pos:
                return args[pos]
            return default

        num_embeddings = _arg(0, "num_embeddings")
        embedding_dim = _arg(1, "embedding_dim")
        quant_config = _arg(5, "quant_config")
        prefix = str(_arg(6, "prefix", "") or "")

        prefix_is_ngram = "ngram_embedding" in prefix.lower()

        group = _tq_manifest_unique_ngram_group(manifest) if manifest is not None else None
        expected_ngram_dim = None
        if manifest is not None and group is not None:
            # Derive M from checkpoint metadata; never assume a model-specific
            # hidden/embedding size such as 160.
            for name in getattr(manifest, "quantized_layers", ()):
                sname = str(name)
                if _tq_ngram_group_name(sname) != group:
                    continue
                try:
                    meta = manifest.load_meta(sname)
                except Exception:
                    continue
                if meta.get("target_kind") != "tq_ngram_chunked_shard":
                    continue
                dim = meta.get("ngram_embedding_dim")
                matrix_shape = meta.get("matrix_shape")
                if dim is None and isinstance(matrix_shape, (list, tuple)) and len(matrix_shape) == 2:
                    dim = matrix_shape[0]
                try:
                    expected_ngram_dim = int(dim)
                except Exception:
                    expected_ngram_dim = None
                break

        try:
            shape_is_ple = (
                expected_ngram_dim is not None
                and int(embedding_dim) == expected_ngram_dim
                and int(num_embeddings) >= 1_000_000
            )
        except Exception:
            shape_is_ple = False

        if not (prefix_is_ngram or shape_is_ple):
            group = None

        if active is not None and manifest is not None and group is not None:
            # Prefer the model-provided prefix when it is already the logical
            # n-gram path. Otherwise use the unique manifest group so TQ's
            # grouped resolver has an exact key.
            effective_prefix = prefix if prefix_is_ngram else group
            a = list(args)
            if len(a) > 5:
                a[5] = active
            else:
                kwargs["quant_config"] = active
            if len(a) > 6:
                a[6] = effective_prefix
            else:
                kwargs["prefix"] = effective_prefix
            args = tuple(a)
            print(
                "[TQ PLE HOOK]",
                "runtime_build=", _TQ_RUNTIME_BUILD,
                "num_embeddings=", num_embeddings,
                "embedding_dim=", embedding_dim,
                "original_prefix=", repr(prefix),
                "effective_prefix=", repr(effective_prefix),
                "original_quant_config=", type(quant_config).__name__ if quant_config is not None else None,
                flush=True,
            )

        return original(self, *args, **kwargs)

    _tq_vocab_embedding_init._tq_ple_hook = True
    _tq_vocab_embedding_init._tq_original = original
    VocabParallelEmbedding.__init__ = _tq_vocab_embedding_init

_install_tq_ple_vocab_embedding_hook()

# MoE quantization support.
#
# FusedMoE / FusedMoEMethodBase have moved slightly across vLLM releases,
# so keep the imports tolerant to both layouts.
try:
    from vllm.model_executor.layers.fused_moe import RoutedExperts
except ImportError:
    try:
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    except ImportError:
        RoutedExperts = ()  # older vLLM: dispatch only through FusedMoE

from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)


def rand_signs_splitmix_fixed(n: int, device: torch.device, seed: int, out_dtype: torch.dtype = torch.float16 ) -> torch.Tensor:
    # SplitMix64 constants as signed int64 (same bit patterns)
    C1 = -7046029254386353131  # 0x9E3779B97F4A7C15
    C2 = -4658895280553007687  # 0xBF58476D1CE4E5B9
    C3 = -7723592293110705685  # 0x94D049BB133111EB
    i = torch.arange(n, dtype=torch.int64, device=device)

    # SplitMix64 on (i ^ FIXED_SEED_D), using scalar constants (no tensor allocs)
    x = (i ^ seed) + C1
    z = x
    z = z ^ (z >> 30)
    z = (z * C2)
    z = z ^ (z >> 27)
    z = (z * C3)
    z = z ^ (z >> 31)

    # LSB -> {0,1} then map to {-1,+1} via (1 - 2*b)
    b = (z & 1).to(out_dtype)        # 0 or 1
    signs = b.neg().mul_(2).add_(1)  # 0->+1, 1->-1
    return signs

# ---------------------------------------------------------------------------
# FFT sizes shared by Group A and Group B
# ---------------------------------------------------------------------------

IFFT_SIZES = {
    4, 8, 12, 16, 20, 24, 32, 36, 40, 48, 60, 64,
    72, 80, 96, 100, 108, 112, 120, 128, 144, 160, 176, 180,
    192, 196, 200, 208, 216, 224, 240, 256, 272, 288, 300, 304,
    320, 324, 336, 352, 360, 368, 384, 400, 416, 432, 448, 464,
    480, 496, 500, 512, 576, 900, 1000, 1024, 1296, 1728, 2048,
    4096, 5832, 7776, 8000, 8192, 10000, 13824, 16384,
}

FFT_SIZES = {
    16, 32, 48, 64, 80, 96, 112, 128,
    144, 160, 176, 192, 208, 224, 240, 256,
    272, 288, 304, 320, 336, 352, 368, 384,
    400, 416, 432, 448, 464, 480, 496, 512,
    576, 1024, 1296, 1728, 2048, 4096,
    7776, 8000, 8192, 10000, 13824, 16384,
}

def can_use_tq_cufftdx(N: int, SIZES) -> bool:
    if (N % 8) != 0:
        return False

    fft_size = N // 2

    if fft_size in SIZES:
        return True

    return False


def _tq_semantic_namespace(name: str) -> str:
    """Classify model subtrees that must never be crossed by fuzzy matching.

    Wrapper prefixes (model., language_model.) are intentionally ignored, but
    semantic submodels such as MTP, vision, and audio are kept distinct.

    This prevents, for example:
      runtime  language_model.model.layers.0.self_attn.qkv_proj
    from incorrectly matching:
      manifest model.mtp.layers.0.self_attn.qkv_proj
    merely because their trailing suffix is unique.
    """
    s = str(name).strip(".")

    # Strip only non-semantic wrapper prefixes.
    changed = True
    while changed:
        changed = False
        for p in ("model.language_model.", "language_model.", "model."):
            if s.startswith(p):
                s = s[len(p):]
                changed = True
                break

    parts = s.split(".")
    if "mtp" in parts:
        return "mtp"
    if parts and parts[0] in ("visual", "vision", "vision_model"):
        return "vision"
    if parts and parts[0] in ("audio", "audio_model", "audio_tower"):
        return "audio"
    return "backbone"


def _tq_same_semantic_namespace(a: str, b: str) -> bool:
    return _tq_semantic_namespace(a) == _tq_semantic_namespace(b)


def resolve_vllm_name(
    prefix: str,
    quantized_layers: set[str] | frozenset[str],
) -> str | None:
    """
    Resolve a native-vLLM runtime module prefix to the corresponding
    TQ manifest/checkpoint layer name.

    This is intentionally architecture-agnostic.

    It does NOT contain model-specific rules for:
        - Qwen
        - Llama
        - Mistral
        - Gemma
        - multimodal models
        - vision towers
        - MoE models

    Instead, the TQ manifest is treated as the source of truth.

    Returns:
        Exact manifest layer name if a unique match can be found.
        None if no match exists.

    Raises:
        RuntimeError if normalization would produce an ambiguous match.
    """

    if not prefix:
        return None

    # ------------------------------------------------------------
    # 1. Exact match
    #
    # This should always be preferred. For most native-vLLM models
    # the runtime namespace and checkpoint namespace already agree.
    # ------------------------------------------------------------
    if prefix in quantized_layers:
        return prefix

    # ------------------------------------------------------------
    # 2. Common wrapper variants
    #
    # These are namespace wrappers, NOT architecture names.
    #
    # Example:
    #
    #   runtime:
    #       visual.blocks.0.attn.qkv
    #
    #   manifest:
    #       model.visual.blocks.0.attn.qkv
    #
    # or:
    #
    #   runtime:
    #       language_model.model.layers.0.mlp.down_proj
    #
    #   manifest:
    #       model.language_model.model.layers.0.mlp.down_proj
    # ------------------------------------------------------------

    candidates = []

    def add(name: str) -> None:
        if name and name not in candidates:
            candidates.append(name)

    # Add/remove one common model wrapper.
    add("model." + prefix)

    if prefix.startswith("model."):
        add(prefix[len("model."):])

    # Some wrappers can be nested by model containers.
    add("language_model." + prefix)
    add("model.language_model." + prefix)

    if prefix.startswith("language_model."):
        rest = prefix[len("language_model."):]
        add(rest)
        add("model." + prefix)
        add("model." + rest)

    if prefix.startswith("model.language_model."):
        rest = prefix[len("model.language_model."):]
        add(rest)
        add("language_model." + rest)
        add("model." + rest)

    direct_matches = [
        candidate
        for candidate in candidates
        if candidate in quantized_layers
        and _tq_same_semantic_namespace(prefix, candidate)
    ]

    if len(direct_matches) == 1:
        return direct_matches[0]

    if len(direct_matches) > 1:
        raise RuntimeError(
            "TQ ambiguous native-vLLM layer resolution:\n"
            f"  runtime prefix: {prefix!r}\n"
            f"  matches: {direct_matches!r}"
        )

    # ------------------------------------------------------------
    # 3. Suffix matching
    #
    # Last resort for wrapper differences we don't know about.
    #
    # We ONLY accept a UNIQUE suffix match.
    #
    # Example:
    #
    # runtime:
    #   decoder.model.layers.5.mlp.down_proj
    #
    # manifest:
    #   some_wrapper.decoder.model.layers.5.mlp.down_proj
    #
    # We progressively require a meaningful suffix rather than
    # matching something dangerously generic like "down_proj".
    # ------------------------------------------------------------

    parts = prefix.split(".")

    # Require at least 3 components.
    #
    # Never resolve merely on:
    #   q_proj
    #   mlp.down_proj
    #
    # because hundreds of layers may contain those names.
    for n_parts in range(len(parts), 2, -1):

        suffix = ".".join(parts[-n_parts:])
        suffix_token = "." + suffix

        matches = [
            name
            for name in quantized_layers
            if _tq_same_semantic_namespace(prefix, name)
            and (name == suffix or name.endswith(suffix_token))
        ]

        if len(matches) == 1:
            return matches[0]

        # If several manifest layers share this suffix, continue.
        # A shorter suffix will only become less specific, however,
        # so once ambiguity appears there's no reason to accept it.
        if len(matches) > 1:
            break

    return None



def _ignore_hf_weight_loader(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    *args,
    **kwargs,
):
    """
    Consume the original HF .weight for an TQ-covered layer.

    The actual matrix is represented by the TQ packed parameters, so there
    is intentionally nothing to copy.
    """
    return


def _register_hf_weight_sink(
    layer: torch.nn.Module,
    extra_weight_attrs: dict | None = None,
) -> None:
    """Install a zero-storage sink for HF weights owned by TQ.

    Native vLLM packed linears (QKV, gate/up, Qwen GDN qkvz/ba, ...) route
    several checkpoint tensors into one runtime ``weight`` parameter and pass a
    ``shard_id`` to that parameter's loader.  A TQ-owned matrix must consume
    those checkpoint tensors without asking vLLM to slice/copy them, because
    the real matrix lives in the TQ packed parameters.

    Keep any harmless attrs supplied by the vLLM LinearBase constructor, but
    ALWAYS override ``weight_loader`` last.  This is important on the
    qwen38-flash-next branch, whose merged-column loader otherwise attempts to
    narrow a zero-sized residual tensor and fails with e.g.
    ``length (...) exceeds dimension size (0)``.
    """
    weight_sink = Parameter(
        torch.empty(0, device=_tq_runtime_device(), dtype=torch.uint8),
        requires_grad=False,
    )

    attrs = dict(extra_weight_attrs or {})
    attrs["ignore_warning"] = True
    attrs["weight_loader"] = _ignore_hf_weight_loader
    set_weight_attrs(weight_sink, attrs)

    # Be explicit as well as using set_weight_attrs.  Some vLLM development
    # branches inspect the attribute directly in AutoWeightsLoader paths.
    weight_sink.weight_loader = _ignore_hf_weight_loader
    weight_sink.ignore_warning = True

    layer.register_parameter("weight", weight_sink)


@dataclass(frozen=True)
class _TqExpertRef:
    group: str
    expert_id: int
    raw_projection: str
    layer_name: str


def _tq_normalize_path(name: str) -> str:
    """Canonicalize equivalent HF/vLLM module paths for manifest lookup."""
    s = name.strip(".")
    for p in ("model.language_model.", "language_model.", "model."):
        if s.startswith(p):
            s = s[len(p):]
            break
    # architecture aliases used by the attached checkpoints
    s = s.replace(".self_attn.", ".attn.")
    s = s.replace(".ffn.", ".mlp.")
    if s.endswith(".ffn"):
        s = s[:-4] + ".mlp"
    s = s.replace(".shared_experts.", ".shared_expert.")
    if s.endswith(".shared_experts"):
        s = s[:-len(".shared_experts")] + ".shared_expert"
    # DeepSeek-style checkpoint projection names -> HF-style semantic names
    repl = {
        ".wq_a": ".q_a_proj", ".wq_b": ".q_b_proj",
        ".wkv": ".kv_proj", ".wo_a": ".o_a_proj", ".wo_b": ".o_b_proj",
        ".w1": ".gate_proj", ".w2": ".down_proj", ".w3": ".up_proj",
    }
    for a, b in repl.items():
        if s.endswith(a):
            s = s[:-len(a)] + b
    # RoutedExperts prefixes vary: some end at mlp/moe/mixer, some at .experts.
    for suffix in (".routed_experts", ".experts", ".expert"):
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    return s.strip(".")


def _tq_parse_expert_layer_name(name: str) -> _TqExpertRef | None:
    """Parse all expert naming layouts emitted by tqizer.py.

    Supported examples:
      model.layers.0.mlp.experts.7.gate_proj
      layers.0.ffn.experts.7.w1
      model.layers.0.mlp.experts.gate_up_proj.expert_0007
      model.layers.3.moe.down_proj.expert_0007
    """
    pats = (
        re.compile(
            r"^(?P<group>.+?)\.experts?\.(?P<eid>\d+)\."
            r"(?P<proj>gate_proj|up_proj|down_proj|gate_up_proj|w1|w2|w3)$"
        ),
        re.compile(
            r"^(?P<group>.+?\.experts)\."
            r"(?P<proj>gate_proj|up_proj|down_proj|gate_up_proj|w1|w2|w3)"
            r"\.expert_(?P<eid>\d+)$"
        ),
        re.compile(
            r"^(?P<group>.+?\.moe)\."
            r"(?P<proj>gate_proj|up_proj|down_proj|gate_up_proj|w1|w2|w3)"
            r"\.expert_(?P<eid>\d+)$"
        ),
        # Generic projection-before-expert fallback, restricted to MoE-ish paths.
        re.compile(
            r"^(?P<group>.+?)\."
            r"(?P<proj>gate_proj|up_proj|down_proj|gate_up_proj|w1|w2|w3)"
            r"\.expert_(?P<eid>\d+)$"
        ),
    )
    for pat in pats:
        m = pat.match(name)
        if not m:
            continue
        gd = m.groupdict()
        group = gd["group"]
        if pat is pats[-1] and not any(tok in group for tok in ("moe", "expert", "ffn", "mlp", "mixer")):
            continue
        return _TqExpertRef(
            group=group,
            expert_id=int(gd["eid"]),
            raw_projection=gd["proj"],
            layer_name=name,
        )
    return None


def _tq_name_role(raw_projection: str) -> str:
    p = raw_projection.lower()
    if p in ("gate_proj", "w1"):
        return "gate"
    if p in ("up_proj", "w3"):
        return "up"
    if p in ("down_proj", "w2"):
        return "down"
    if p in ("gate_up_proj", "w13"):
        return "gate_up"
    raise ValueError(raw_projection)


def _tq_record_role(
    raw_projection: str,
    matrix_shape: tuple[int, int],
    hidden_size: int,
    intermediate_size: int,
) -> str:
    """Infer semantic role from BOTH name and matrix shape.

    Shape inference is important for architectures such as Nemotron where a
    checkpoint may call the fused SwiGLU first projection `up_proj`.
    """
    if len(matrix_shape) != 2:
        raise RuntimeError(f"TQ expert matrix must be 2-D, got {matrix_shape}")
    out_f, in_f = map(int, matrix_shape)
    H, I = int(hidden_size), int(intermediate_size)

    # tolerate a transposed source only for classification
    if (out_f, in_f) == (2 * I, H) or (out_f, in_f) == (H, 2 * I):
        return "gate_up"
    if (out_f, in_f) == (H, I) or (out_f, in_f) == (I, H):
        named = _tq_name_role(raw_projection)
        if named == "down" or (out_f, in_f) == (H, I):
            return "down"
        return named

    # Fall back to the explicit name if the model uses a nonstandard dimension.
    return _tq_name_role(raw_projection)


def _tq_orient_matrix(w: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    if tuple(w.shape) == (out_features, in_features):
        return w
    if tuple(w.shape) == (in_features, out_features):
        return w.t().contiguous()
    raise RuntimeError(
        f"Cannot orient expert matrix {tuple(w.shape)} as "
        f"({out_features}, {in_features})"
    )


def _tq_routed_experts_load_weights(layer, weights):
    """Instance-level RoutedExperts loader installed by TqMoEMethod.

    The custom method owns expert checkpoint loading. Quantized expert matrices
    are consumed/discarded here (their packed data comes from quantization_data),
    while missing matrices are copied into sparse dense fallback buffers.
    Anything not recognized as an expert model weight is delegated to vLLM's
    original RoutedExperts.load_weights implementation.
    """
    method = getattr(layer, "quant_method", None)
    if not isinstance(method, TqMoEMethod):
        yield from layer._tq_original_load_weights(weights)
        return

    leftovers = []
    for name, tensor in weights:
        if method._consume_hf_expert_weight(layer, name, tensor):
            # AutoWeightsLoader only needs an iterable of loaded names.
            yield f"tq_consumed:{name}"
        else:
            leftovers.append((name, tensor))

    if leftovers:
        yield from layer._tq_original_load_weights(leftovers)


class TqModelManifest:

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)

        # The quantizer writes "config.json"; some directories are hand-named
        # "conf.json" instead. Accept either, preferring config.json.
        config_candidates = [
            self.model_dir / "config.json",
            self.model_dir / "conf.json",
        ]
        self.config_path = next(
            (p for p in config_candidates if p.exists()),
            config_candidates[0],
        )
        self.quant_dir = self.model_dir / "quantization_data"
        self.layers_path = self.quant_dir / "_layers.json"

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Missing TQ config: tried {[str(p) for p in config_candidates]}"
            )

        if not self.layers_path.exists():
            raise FileNotFoundError(
                f"Missing TQ layer manifest: {self.layers_path}"
            )

        with open(self.config_path, "r") as f:
            self.hf_config = json.load(f)

        with open(self.layers_path, "r") as f:
            layer_names = json.load(f)

        self.quantized_layers = frozenset(layer_names)

        # ------------------------------------------------------------------
        # Preferred v2 storage: a small number of large TQ shard files in the
        # same HF repo as the residual checkpoint. Each layer maps to one shard.
        # ------------------------------------------------------------------
        self.shard_index_path = self.quant_dir / "tq_shards.json"
        self.shard_repo_id: str | None = None
        self.shard_layer_locations: dict[str, dict] = {}
        self.shard_format: str | None = None

        if self.shard_index_path.exists():
            with open(self.shard_index_path, "r", encoding="utf-8") as f:
                shard_index = json.load(f)

            self.shard_format = shard_index.get("format")
            if self.shard_format not in (
                "tq_shards_v1",
                "tq_safetensors_shards_v1",
            ):
                raise RuntimeError(
                    f"Unsupported TQ shard index format in "
                    f"{self.shard_index_path}: {self.shard_format!r}"
                )

            self.shard_repo_id = shard_index.get("repo_id")
            self.shard_layer_locations = {
                str(k): dict(v)
                for k, v in shard_index.get("layers", {}).items()
            }

        # Backward compatibility with the old one-file-per-layer / multi-repo
        # uploader. New uploads should use tq_shards.json instead.
        self.remote_index_path = self.model_dir / "quantization_repos.json"
        self.remote_layer_locations: dict[str, dict[str, str]] = {}
        if self.remote_index_path.exists():
            with open(self.remote_index_path, "r", encoding="utf-8") as f:
                remote_index = json.load(f)
            if remote_index.get("format") != "tq_layer_repos_v1":
                raise RuntimeError(
                    f"Unsupported TQ remote index format in "
                    f"{self.remote_index_path}: {remote_index.get('format')!r}"
                )
            self.remote_layer_locations = {
                str(k): dict(v)
                for k, v in remote_index.get("layers", {}).items()
            }

        # torch.load() deserializes an entire .pt shard. Keep only a small LRU
        # of deserialized shards so consecutive records reuse one load without
        # pinning the whole quantized model in CPU RAM.
        from collections import OrderedDict
        self._shard_cache = OrderedDict()
        self._shard_cache_lock = threading.Lock()
        self._shard_cache_size = max(
            1, int(os.environ.get("TQ_SHARD_CACHE_SIZE", "1"))
        )

        self._build_manifest_indexes()

        self.base_model = self.hf_config.get("_name_or_path")

        if not self.base_model:
            raise RuntimeError(
                f"{self.config_path} has no _name_or_path"
            )

    def __getstate__(self):
        """
        Make the manifest safe for multiprocessing spawn.

        vLLM pickles VllmConfig when spawning its EngineCore worker.
        threading.Lock is not pickleable, and deserialized shard contents
        should not be copied into the child process anyway.
        """
        state = self.__dict__.copy()

        state.pop(
            "_shard_cache_lock",
            None,
        )

        # Don't serialize potentially huge loaded shards.
        state["_shard_cache"] = None

        return state


    def __setstate__(self, state):
        """
        Recreate process-local runtime state after unpickling.
        """
        self.__dict__.update(state)

        from collections import OrderedDict

        self._shard_cache = OrderedDict()

        self._shard_cache_lock = threading.Lock()

    def resolve_runtime_name(self, prefix: str) -> str | None:
        return resolve_vllm_name(
            prefix,
            self.quantized_layers,
        )
    
    @staticmethod
    def sanitize_layer_name(layer_name: str) -> str:
        return (
            layer_name
            .replace(".", "_")
            .replace("/", "_")
        )

    def layer_file(self, layer_name: str) -> Path:
        """Legacy one-file-per-layer resolver."""
        safe_name = self.sanitize_layer_name(layer_name)
        local_path = self.quant_dir / f"{safe_name}.pt"

        if local_path.exists():
            return local_path

        location = self.remote_layer_locations.get(layer_name)
        if location is None:
            return local_path

        repo_id = location.get("repo_id")
        filename = location.get("filename")
        if not repo_id or not filename:
            raise RuntimeError(
                f"Invalid remote location for TQ layer {layer_name!r}: "
                f"{location!r}"
            )

        from huggingface_hub import hf_hub_download

        try:
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                token=os.environ.get("HF_TOKEN"),
                local_files_only=True,
            )
            return Path(cached_path)
        except Exception:
            pass

        _tq_log(
            f"legacy layer cache miss, fetching over network: "
            f"repo={repo_id} file={filename} layer={layer_name!r}"
        )

        cached_path = _tq_run_with_timeout(
            lambda: hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                token=os.environ.get("HF_TOKEN"),
            ),
            desc=f"hf_hub_download(repo={repo_id!r}, file={filename!r})",
        )
        return Path(cached_path)

    def _shard_file(self, layer_name: str) -> Path:
        location = self.shard_layer_locations.get(layer_name)
        if location is None:
            raise KeyError(layer_name)

        filename = location.get("filename")
        if not filename:
            raise RuntimeError(
                f"Invalid TQ shard location for {layer_name!r}: {location!r}"
            )

        # For a fully materialized local model directory / Endpoint download,
        # the shard is already beside the residual checkpoint.
        local_path = self.model_dir / filename
        if local_path.exists():
            return local_path

        # For a repo ID, get_active_tq_manifest() initially snapshots only the
        # small metadata files. Fetch each large shard on first use. HF/Xet then
        # keeps the shard in the normal Hub cache.
        repo_id = location.get("repo_id") or self.shard_repo_id
        if not repo_id:
            raise RuntimeError(
                f"TQ shard {filename!r} is not local and the shard index has "
                f"no repo_id for layer {layer_name!r}"
            )

        from huggingface_hub import hf_hub_download

        # Try the local Hub cache first with local_files_only=True. This
        # never touches the network -- it just resolves the on-disk cache
        # path. hf_hub_download's normal (online) path still performs an
        # HTTP etag/HEAD check even when the file is already fully cached;
        # if that metadata round-trip stalls (rate limiting, DNS, a flaky
        # proxy) it hangs this worker even though nothing actually needed
        # to be downloaded. This also avoids one redundant HTTP call per
        # expert/layer that shares an already-downloaded shard file.
        try:
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                token=os.environ.get("HF_TOKEN"),
                local_files_only=True,
            )
            return Path(cached_path)
        except Exception:
            pass

        _tq_log(
            f"shard cache miss, fetching over network: "
            f"repo={repo_id} file={filename} layer={layer_name!r}"
        )

        cached_path = _tq_run_with_timeout(
            lambda: hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                token=os.environ.get("HF_TOKEN"),
            ),
            desc=f"hf_hub_download(repo={repo_id!r}, file={filename!r})",
        )
        return Path(cached_path)

    def _load_safetensor_tensors(self, layer_name: str) -> dict:
        """Load only this layer's tensors from a safetensors shard."""
        location = self.shard_layer_locations.get(layer_name)
        if location is None:
            raise KeyError(layer_name)

        tensor_keys = location.get("tensor_keys")
        if not isinstance(tensor_keys, dict):
            raise RuntimeError(
                f"TQ safetensors index entry for {layer_name!r} has no tensor_keys"
            )

        path = self._shard_file(layer_name)

        try:
            from safetensors import safe_open
        except ImportError as e:
            raise RuntimeError(
                "safetensors is required to load TQ safetensors shards"
            ) from e

        out = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for logical_name, storage_name in tensor_keys.items():
                out[str(logical_name)] = f.get_tensor(str(storage_name))
        return out

    def _load_shard(self, layer_name: str) -> dict:
        path = self._shard_file(layer_name)
        cache_key = str(path)

        with self._shard_cache_lock:
            cached = self._shard_cache.get(cache_key)
            if cached is not None:
                self._shard_cache.move_to_end(cache_key)
                return cached

        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(obj, dict) or obj.get("format") != "tq_shard_v1":
            raise RuntimeError(
                f"Invalid TQ shard file {path}: expected format='tq_shard_v1'"
            )

        records = obj.get("records")
        if not isinstance(records, dict):
            raise RuntimeError(f"Invalid TQ shard file {path}: missing records dict")

        with self._shard_cache_lock:
            self._shard_cache[cache_key] = records
            self._shard_cache.move_to_end(cache_key)
            while len(self._shard_cache) > self._shard_cache_size:
                self._shard_cache.popitem(last=False)

        return records

    def _build_manifest_indexes(self) -> None:
        self._expert_groups: dict[str, dict[int, list[_TqExpertRef]]] = {}
        self._expert_group_norm: dict[str, list[str]] = {}
        self._linear_norm: dict[str, list[str]] = {}

        for name in self.quantized_layers:
            ref = _tq_parse_expert_layer_name(name)
            if ref is not None:
                self._expert_groups.setdefault(ref.group, {}).setdefault(
                    ref.expert_id, []
                ).append(ref)
                norm = _tq_normalize_path(ref.group)
                self._expert_group_norm.setdefault(norm, []).append(ref.group)
            else:
                norm = _tq_normalize_path(name)
                self._linear_norm.setdefault(norm, []).append(name)

        # deduplicate group lists while keeping deterministic order
        self._expert_group_norm = {
            k: sorted(set(v)) for k, v in self._expert_group_norm.items()
        }
        self._linear_norm = {
            k: sorted(set(v)) for k, v in self._linear_norm.items()
        }

    def resolve_linear_name(self, prefix: str) -> str | None:
        if prefix in self.quantized_layers and _tq_parse_expert_layer_name(prefix) is None:
            return prefix
        norm = _tq_normalize_path(prefix)
        candidates = self._linear_norm.get(norm, [])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise RuntimeError(
                f"TQ manifest: ambiguous linear alias for {prefix!r}: {candidates}"
            )
        return None

    def resolve_packed_linear(
        self,
        prefix: str,
    ) -> tuple[str, list[str]] | None:
        """Resolve native-vLLM packed linear modules to HF/TQ records.

        Offline TQ records keep Hugging Face checkpoint granularity, while
        native vLLM packs several source projections into one runtime module.
        This resolver bridges those two representations without changing the
        quantizer/checkpoint format.
        """

        # Keep this table in lock-step with tq_quantizer.py's
        # NATIVE_VLLM_PACKED_LINEAR_RULES.  New checkpoints are normally
        # quantized under the native fused target name itself; these rules are
        # still required for older/split TQ manifests and for wrapper-name
        # differences.
        rules = (
            ("qkv_proj", "qkv", ("q_proj", "k_proj", "v_proj")),
            ("gate_up_proj", "gate_up", ("gate_proj", "up_proj")),
            ("in_proj_qkvz", "qkvz", ("in_proj_qkv", "in_proj_z")),
            ("in_proj_ba", "ba", ("in_proj_b", "in_proj_a")),
        )

        for packed_suffix, kind, source_suffixes in rules:
            token = f".{packed_suffix}"
            if not prefix.endswith(token):
                continue

            base = prefix[:-len(token)]
            resolved: list[str] = []
            for source_suffix in source_suffixes:
                source_name = f"{base}.{source_suffix}"
                manifest_name = self.resolve_linear_name(source_name)
                if manifest_name is None:
                    return None
                resolved.append(manifest_name)

            return kind, resolved

        return None

    def resolve_expert_group(
        self, moe_prefix: str
    ) -> tuple[str, dict[int, list[_TqExpertRef]]] | None:
        norm = _tq_normalize_path(moe_prefix)
        groups = self._expert_group_norm.get(norm, [])

        if not groups:
            # Fallback to suffix matching. This handles wrapper prefixes added by
            # Transformers/vLLM without making model-specific assumptions.
            matches = []
            for n, gs in self._expert_group_norm.items():
                if n.endswith(norm) or norm.endswith(n):
                    matches.extend(gs)
            groups = sorted(set(matches))

        if len(groups) == 1:
            g = groups[0]
            return g, self._expert_groups[g]
        if len(groups) > 1:
            raise RuntimeError(
                f"TQ manifest: ambiguous MoE alias for {moe_prefix!r} "
                f"(normalized={norm!r}): {groups}"
            )
        return None

    def has_quantized_experts(self, moe_prefix: str) -> bool:
        return self.resolve_expert_group(moe_prefix) is not None

    # Backward-compatible helper. For nonstandard layouts prefer
    # resolve_expert_group(), which carries the actual record names.


    def load_record(
        self,
        layer_name: str,
    ) -> dict:
        """Return one logical TQ record, independent of physical storage."""
        if layer_name in self.shard_layer_locations:
            if self.shard_format == "tq_safetensors_shards_v1":
                return {
                    "tensors": self._load_safetensor_tensors(layer_name),
                    "meta": self.load_meta(layer_name),
                }

            # Legacy .pt shard layout.
            records = self._load_shard(layer_name)
            try:
                return records[layer_name]
            except KeyError as e:
                path = self._shard_file(layer_name)
                raise KeyError(
                    f"TQ shard index maps {layer_name!r} to {path}, but the "
                    f"record is absent from that shard"
                ) from e

        # Backward-compatible local / old multi-repo one-file-per-layer layout.
        path = self.layer_file(layer_name)
        if not path.exists():
            raise FileNotFoundError(
                f"TQ layer is listed in _layers.json but no record was found:\n"
                f"  layer={layer_name}\n"
                f"  file={path}"
            )
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    def load_tensors(
        self,
        layer_name: str,
    ) -> dict:
        # Safetensors path: only map/read the three tensors required by this
        # logical matrix (packed_all, u_W, std_W). No whole-shard torch.load.
        if (
            self.shard_format == "tq_safetensors_shards_v1"
            and layer_name in self.shard_layer_locations
        ):
            return self._load_safetensor_tensors(layer_name)

        return self.load_record(layer_name)["tensors"]

    def load_meta(
        self,
        layer_name: str,
    ) -> dict:
        # Shard metadata is duplicated into the JSON index, so MoE planning
        # never needs to deserialize/map a large shard just to inspect shapes.
        location = self.shard_layer_locations.get(layer_name)
        if location is not None and "meta" in location:
            return location["meta"]
        return self.load_record(layer_name)["meta"]


# Cache the active manifest within each worker process.
#
# vLLM 0.27.1 passes the exact model argument (local directory or Hub repo ID)
# to QuantizationConfig.maybe_update_config(). TQ_MODEL_DIR is therefore only
# an optional explicit override, not a required customer-facing setting.

_ACTIVE_TQ_MANIFEST = None
_ACTIVE_TQ_MODEL_REF = None
_ACTIVE_TQ_REVISION = None


# def get_active_tq_manifest(
#     model_ref: str | None = None,
#     *,
#     revision: str | None = None,
# ):
def get_active_tq_manifest(
    model_ref=None,
    *,
    revision=None,
):
    global _ACTIVE_TQ_MANIFEST, _ACTIVE_TQ_MODEL_REF, _ACTIVE_TQ_REVISION

    # Explicit environment override wins when present. This is useful for
    # unusual deployments where the TQ metadata lives somewhere different
    # from the model argument supplied to vLLM.
    model_ref = os.environ.get("TQ_MODEL_DIR") or model_ref

    if not model_ref:
        if _ACTIVE_TQ_MANIFEST is not None:
            return _ACTIVE_TQ_MANIFEST
        raise RuntimeError(
            "TQ could not determine the model location. Pass the model to "
            "vLLM normally (vllm serve MODEL or LLM(model=MODEL)), or set "
            "TQ_MODEL_DIR as an explicit override."
        )

    model_ref = str(model_ref)

    if (
        _ACTIVE_TQ_MANIFEST is not None
        and _ACTIVE_TQ_MODEL_REF == model_ref
        and _ACTIVE_TQ_REVISION == revision
    ):
        return _ACTIVE_TQ_MANIFEST

    model_path = Path(model_ref).expanduser()

    # Local TQ GGUF supplied through TQ_GGUF or directly by internal helpers.
    gguf_override = os.environ.get("TQ_GGUF")
    if gguf_override:
        from .gguf_storage import TqGGUFManifest
        manifest = TqGGUFManifest(gguf_override)
        _ACTIVE_TQ_MANIFEST = manifest
        _ACTIVE_TQ_MODEL_REF = model_ref
        _ACTIVE_TQ_REVISION = revision
        return manifest

    if model_path.exists() and model_path.is_file() and model_path.suffix.lower() == ".gguf":
        from .gguf_storage import TqGGUFManifest
        manifest = TqGGUFManifest(model_path.resolve())
        _ACTIVE_TQ_MANIFEST = manifest
        _ACTIVE_TQ_MODEL_REF = model_ref
        _ACTIVE_TQ_REVISION = revision
        return manifest

    # Local directory supplied directly to vLLM.
    if model_path.exists():
        resolved_dir = model_path.resolve()

    # Hugging Face repo ID supplied directly to vLLM.
    else:
        model_ref = str(model_ref)

        if revision is not None and not isinstance(revision, str):
            revision = getattr(revision, "revision", None) or str(revision)

        _tq_log(f"snapshot_download starting: repo={model_ref} revision={revision}")

        resolved_dir = Path(
            _tq_run_with_timeout(
                lambda: snapshot_download(
                    repo_id=model_ref,
                    repo_type="model",
                    revision=revision,
                    allow_patterns=[
                        "config.json",
                        "conf.json",
                        "quantization_data/_layers.json",
                        "quantization_data/tq_shards.json",
                        "quantization_data/tq-*.safetensors",
                        "quantization_data/tq-*.pt",  # legacy sharded layout
                    ],
                    max_workers=8,
                    token=os.environ.get("HF_TOKEN"),
                ),
                desc=f"snapshot_download(repo={model_ref!r})",
            )
        )

    manifest = TqModelManifest(resolved_dir)

    _ACTIVE_TQ_MANIFEST = manifest
    _ACTIVE_TQ_MODEL_REF = model_ref
    _ACTIVE_TQ_REVISION = revision
    return manifest

@torch.no_grad()
def _load_tq_record_into_packed(
    layer: torch.nn.Module,
    manifest,
    prefix: str,
) -> None:

    tensors = manifest.load_tensors(prefix)

    dst = layer.packed_all
    B = dst.shape[0]

    # New safetensors layout stores the exact final runtime representation.
    if "packed_all" in tensors:
        packed = tensors["packed_all"]
        if tuple(packed.shape) != tuple(dst.shape):
            raise RuntimeError(
                f"TQ packed_all shape mismatch for {prefix}: "
                f"saved={tuple(packed.shape)} runtime={tuple(dst.shape)}"
            )
        dst.copy_(packed.to(device=dst.device, dtype=torch.uint8))
        layer.u_W.copy_(
            tensors["u_W"].to(device=dst.device, dtype=torch.float32).reshape(())
        )
        layer.std_W.copy_(
            tensors["std_W"].to(device=dst.device, dtype=torch.float32).reshape(())
        )
        return

    sig1 = tensors["SigRec1_select_packed"]
    sig2 = tensors["SigRec2_select_packed"]
    sig3 = tensors["SigRec3_select_packed"]
    sig4 = tensors["SigRec4_select_packed"]
    x567 = tensors["X567_packed"]

    assert sig1.shape == (B, 7)
    assert sig2.shape == (B, 12)
    assert sig3.shape == (B, 29)
    assert sig4.shape == (B, 83)
    assert x567.shape == (3 * B, 128)

    # packed_all was zero initialized, so padding bytes
    # are already correct.
    #
    # layout:
    #
    #   0:8       sig1   [7 + 1 pad]
    #   8:20      sig2   [12]
    #  20:52      sig3   [29 + 3 pad]
    #  52:136     sig4   [83 + 1 pad]
    # 136:264     X5
    # 264:392     X6
    # 392:520     X7

    dst[:, 0:7].copy_(
        sig1.to(
            device=dst.device,
            dtype=torch.uint8,
        )
    )

    dst[:, 8:20].copy_(
        sig2.to(
            device=dst.device,
            dtype=torch.uint8,
        )
    )

    dst[:, 20:49].copy_(
        sig3.to(
            device=dst.device,
            dtype=torch.uint8,
        )
    )

    dst[:, 52:135].copy_(
        sig4.to(
            device=dst.device,
            dtype=torch.uint8,
        )
    )

    dst[:, 136:264].copy_(
        x567[:B].to(
            device=dst.device,
            dtype=torch.uint8,
        )
    )

    dst[:, 264:392].copy_(
        x567[B:2 * B].to(
            device=dst.device,
            dtype=torch.uint8,
        )
    )

    dst[:, 392:520].copy_(
        x567[2 * B:3 * B].to(
            device=dst.device,
            dtype=torch.uint8,
        )
    )

    layer.u_W.copy_(
        tensors["u_W"]
        .to(
            device=dst.device,
            dtype=torch.float32,
        )
        .reshape(())
    )

    layer.std_W.copy_(
        tensors["std_W"]
        .to(
            device=dst.device,
            dtype=torch.float32,
        )
        .reshape(())
    )



# ---------------------------------------------------------------------------
# MiMo-V2 fused-QKV compatibility
# ---------------------------------------------------------------------------
#
# Xiaomi MiMo-V2.x "fused_qkv" checkpoints store QKV rows grouped per KV head:
#
#   [Q_1 | K_1 | V_1 | Q_2 | K_2 | V_2 | ...]
#
# Native vLLM QKVParallelLinear expects the runtime output layout:
#
#   [all Q | all K | all V]
#
# TQ bypasses vLLM's normal dense/FP8 weight loader, so TQ must perform that
# row permutation itself.  We do it on the *output activations* after the TQ
# GEMM, which is mathematically equivalent to permuting the weight rows before
# quantization and lets existing TQ checkpoints remain usable.
#
# MiMo MTP depth 0 is also SWA-shaped in the checkpoint. Some vLLM versions
# instantiate model.mtp.layers.0 as if it were backbone layer 0 (full attention),
# producing 13568 rows instead of the correct SWA 14848 rows.  The patch below
# forces only TQ MiMo MTP decoder layers to use the SWA branch.
# ---------------------------------------------------------------------------





# Do NOT monkey-patch MiMo decoder construction at import time.
# vLLM's dedicated MiMo MTP implementation already constructs MTP attention
# with SWA geometry. The TQ manifest resolver must instead keep backbone and
# MTP namespaces distinct.


def _tq_mimo_qkv_layout_spec(
    hf_config: dict,
    prefix: str,
    saved_M: int,
):
    """Return MiMo grouped-QKV geometry if this TQ record needs deinterleaving."""
    if not isinstance(hf_config, dict):
        return None

    if hf_config.get("model_type") != "mimo_v2":
        return None

    if hf_config.get("tq_mimo_qkv_layout") != "checkpoint_grouped":
        return None

    # Language-model fused qkv only. Vision uses a different qkv module/layout.
    if not str(prefix).endswith(".qkv_proj"):
        return None

    def make_spec(kind: str):
        if kind == "swa":
            num_heads = int(hf_config["swa_num_attention_heads"])
            num_kv_heads = int(hf_config["swa_num_key_value_heads"])
            head_dim = int(hf_config["swa_head_dim"])
            v_head_dim = int(hf_config.get("swa_v_head_dim", head_dim))
        else:
            num_heads = int(hf_config["num_attention_heads"])
            num_kv_heads = int(hf_config["num_key_value_heads"])
            head_dim = int(hf_config["head_dim"])
            v_head_dim = int(hf_config.get("v_head_dim", head_dim))

        if num_heads % num_kv_heads != 0:
            raise RuntimeError(
                f"MiMo QKV layout invalid for {prefix!r}: "
                f"num_heads={num_heads}, num_kv_heads={num_kv_heads}"
            )

        q_rows_per_group = (num_heads // num_kv_heads) * head_dim
        rows_per_group = q_rows_per_group + head_dim + v_head_dim
        total_rows = num_kv_heads * rows_per_group

        return {
            "kind": kind,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "v_head_dim": v_head_dim,
            "q_rows_per_group": q_rows_per_group,
            "rows_per_group": rows_per_group,
            "total_rows": total_rows,
        }

    specs = [make_spec("full"), make_spec("swa")]
    matches = [s for s in specs if int(s["total_rows"]) == int(saved_M)]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        # Extremely unlikely, but avoid silently choosing a wrong layout.
        raise RuntimeError(
            f"Ambiguous MiMo QKV geometry for {prefix!r}, saved_M={saved_M}: "
            f"{matches}"
        )

    return None


def _tq_mimo_deinterleave_qkv_output(
    y: torch.Tensor,
    spec: dict,
) -> torch.Tensor:
    """Convert [Q1,K1,V1,Q2,K2,V2,...] -> [all Q, all K, all V]."""
    n_kv = int(spec["num_kv_heads"])
    qpg = int(spec["q_rows_per_group"])
    hd = int(spec["head_dim"])
    vhd = int(spec["v_head_dim"])
    rpg = int(spec["rows_per_group"])
    total = int(spec["total_rows"])

    if y.shape[-1] != total:
        raise RuntimeError(
            f"TQ MiMo QKV output width mismatch: got {y.shape[-1]}, "
            f"expected {total} for {spec['kind']} geometry"
        )

    leading = y.shape[:-1]
    g = y.reshape(*leading, n_kv, rpg)

    q = g[..., :, :qpg].reshape(*leading, n_kv * qpg)
    k = g[..., :, qpg:qpg + hd].reshape(*leading, n_kv * hd)
    v = g[..., :, qpg + hd:qpg + hd + vhd].reshape(
        *leading, n_kv * vhd
    )

    return torch.cat((q, k, v), dim=-1)



# ---------------------------------------------------------------------------
# TQ-aware pipeline-parallel partitioning
# ---------------------------------------------------------------------------

def _tq_cli_int_flag(name: str) -> int | None:
    """Read an integer CLI flag from sys.argv without importing vLLM CLI code."""
    import sys

    argv = list(sys.argv)
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except Exception:
                return None
        prefix = name + "="
        if arg.startswith(prefix):
            try:
                return int(arg[len(prefix):])
            except Exception:
                return None
    return None


def _tq_detect_pp_size() -> int:
    """Best-effort PP-size detection before vLLM constructs model workers."""
    explicit = os.environ.get("TQ_PP_SIZE")
    if explicit:
        try:
            return max(1, int(explicit))
        except Exception:
            pass

    cli = _tq_cli_int_flag("--pipeline-parallel-size")
    if cli is not None:
        return max(1, cli)

    # HF Inference Endpoints normally expose only the GPUs assigned to the
    # replica. For TQ, TP is generally 1 because the FFT transform prevents
    # tensor-parallel sharding, so visible GPU count is a useful final fallback.
    try:
        return max(1, int(torch.cuda.device_count()))
    except Exception:
        return 1


def _tq_tq_storage_bytes_for_matrix(out_f: int, in_f: int) -> int:
    """Approximate resident TQ bytes for one quantized matrix.

    TQ is nominally 4-bit. The packed representation has small metadata/
    alignment overhead, so use an adjustable multiplier rather than pretending
    the exact byte count is always 0.5 bytes/weight.
    """
    overhead = float(os.environ.get("TQ_PP_QUANT_OVERHEAD", "1.08"))
    return int(float(out_f) * float(in_f) * 0.5 * overhead)


def _tq_dense_storage_bytes(out_f: int, in_f: int) -> int:
    # Runtime is normally fp16/bf16 for MiMo. Keep this configurable.
    dense_bytes = int(os.environ.get("TQ_PP_DENSE_BYTES_PER_ELEM", "2"))
    return int(out_f) * int(in_f) * dense_bytes


def _tq_estimate_transformer_layer_bytes(
    manifest: "TqModelManifest",
    layer_idx: int,
) -> int:
    """Estimate actual GPU weight bytes of one transformer layer.

    The dominant MiMo term is routed-expert storage. Exact quantized
    expert/projection coverage comes from _layers.json; missing projections are
    charged at dense fp16/bf16 cost because TqMoEMethod must retain them as
    fallback weights.

    Non-expert transformer weights are charged conservatively as a small
    per-layer base. This is sufficient for selecting a much better PP cut while
    avoiding heavyweight shard reads during process startup.
    """
    cfg = manifest.hf_config

    H = int(cfg.get("hidden_size", 0) or 0)
    I = int(cfg.get("moe_intermediate_size", 0) or 0)
    E = int(cfg.get("n_routed_experts", 0) or 0)

    total = 0

    # Approximate attention/norm/router/shared small weights. This term is tiny
    # relative to a 256-expert MiMo layer but prevents zero-cost dense layers.
    base_mib = float(os.environ.get("TQ_PP_LAYER_BASE_MIB", "96"))
    total += int(base_mib * 1024 * 1024)

    if not (H and I and E):
        return total

    resolved = manifest.resolve_expert_group(f"model.layers.{layer_idx}.mlp")
    by_expert = resolved[1] if resolved is not None else {}

    for expert_id in range(E):
        refs = by_expert.get(expert_id, [])
        roles = set()
        for ref in refs:
            try:
                roles.add(_tq_name_role(ref.raw_projection))
            except Exception:
                continue

        # gate_up is one fused TQ record that covers both source projections.
        if "gate_up" in roles:
            total += _tq_tq_storage_bytes_for_matrix(2 * I, H)
        else:
            if "gate" in roles:
                total += _tq_tq_storage_bytes_for_matrix(I, H)
            else:
                total += _tq_dense_storage_bytes(I, H)

            if "up" in roles:
                total += _tq_tq_storage_bytes_for_matrix(I, H)
            else:
                total += _tq_dense_storage_bytes(I, H)

        if "down" in roles:
            total += _tq_tq_storage_bytes_for_matrix(H, I)
        else:
            total += _tq_dense_storage_bytes(H, I)

    return total


def _tq_choose_contiguous_pp_partition(
    layer_bytes: list[int],
    pp_size: int,
    *,
    stage0_extra_bytes: int = 0,
    last_stage_extra_bytes: int = 0,
) -> list[int]:
    """Choose contiguous PP layer counts minimizing maximum estimated stage HBM.

    vLLM pipeline stages must own contiguous hidden-layer ranges. Dynamic
    programming finds the minimum possible max stage weight for any PP size.
    Fixed multimodal/input overhead may be charged to rank 0; output-head
    overhead may be charged to the final rank.
    """
    n = len(layer_bytes)
    if pp_size <= 1:
        return [n]
    if pp_size > n:
        raise ValueError(
            f"TQ auto PP partition cannot split {n} layers over {pp_size} ranks"
        )

    prefix = [0]
    for b in layer_bytes:
        prefix.append(prefix[-1] + int(b))

    INF = 1 << 120
    dp = [[INF] * (n + 1) for _ in range(pp_size + 1)]
    prev = [[-1] * (n + 1) for _ in range(pp_size + 1)]
    dp[0][0] = 0

    for stages in range(1, pp_size + 1):
        # at least one layer per stage
        for end in range(stages, n + 1):
            for cut in range(stages - 1, end):
                if dp[stages - 1][cut] >= INF:
                    continue
                stage_bytes = prefix[end] - prefix[cut]

                stage_index = stages - 1
                if stage_index == 0:
                    stage_bytes += int(stage0_extra_bytes)

                # Only charge output-head/final fixed cost if this is the final
                # PP stage and it reaches the final transformer layer.
                if stages == pp_size and end == n:
                    stage_bytes += int(last_stage_extra_bytes)

                score = max(dp[stages - 1][cut], stage_bytes)
                if score < dp[stages][end]:
                    dp[stages][end] = score
                    prev[stages][end] = cut

    counts = []
    end = n
    for stages in range(pp_size, 0, -1):
        cut = prev[stages][end]
        if cut < 0:
            raise RuntimeError("TQ auto PP partition DP failed")
        counts.append(end - cut)
        end = cut

    counts.reverse()
    return counts


_TQ_AUTO_PP_DONE = False


def configure_tq_pipeline_partition(
    manifest: "TqModelManifest | None" = None,
) -> list[int] | None:
    """Set VLLM_PP_LAYER_PARTITION from actual TQ expert coverage.

    Must run before vLLM workers construct model layers. An explicit
    VLLM_PP_LAYER_PARTITION always wins and is never overwritten.
    """
    global _TQ_AUTO_PP_DONE

    if _TQ_AUTO_PP_DONE:
        existing = os.environ.get("VLLM_PP_LAYER_PARTITION")
        return [int(x) for x in existing.split(",")] if existing else None

    auto = os.environ.get("TQ_AUTO_PP_PARTITION", "1").strip().lower()
    if auto in ("0", "false", "no", "off"):
        _TQ_AUTO_PP_DONE = True
        return None

    existing = os.environ.get("VLLM_PP_LAYER_PARTITION")
    if existing:
        _TQ_AUTO_PP_DONE = True
        print(
            "[TQ PP] respecting explicit VLLM_PP_LAYER_PARTITION=",
            existing,
            flush=True,
        )
        return [int(x) for x in existing.split(",")]

    pp_size = _tq_detect_pp_size()
    if pp_size <= 1:
        _TQ_AUTO_PP_DONE = True
        return [int((manifest or get_active_tq_manifest()).hf_config.get(
            "num_hidden_layers", 0
        ) or 0)]

    manifest = manifest or get_active_tq_manifest()
    cfg = manifest.hf_config
    n_layers = int(cfg.get("num_hidden_layers", 0) or 0)
    if n_layers <= 0:
        print("[TQ PP] num_hidden_layers unavailable; leaving vLLM default", flush=True)
        _TQ_AUTO_PP_DONE = True
        return None

    layer_bytes = [
        _tq_estimate_transformer_layer_bytes(manifest, i)
        for i in range(n_layers)
    ]

    stage0_extra_gib = float(os.environ.get("TQ_PP_STAGE0_EXTRA_GIB", "0"))
    last_extra_gib = float(os.environ.get("TQ_PP_LAST_STAGE_EXTRA_GIB", "0"))

    stage0_extra = int(stage0_extra_gib * (1024 ** 3))
    last_extra = int(last_extra_gib * (1024 ** 3))

    counts = _tq_choose_contiguous_pp_partition(
        layer_bytes,
        pp_size,
        stage0_extra_bytes=stage0_extra,
        last_stage_extra_bytes=last_extra,
    )

    os.environ["VLLM_PP_LAYER_PARTITION"] = ",".join(str(x) for x in counts)
    _TQ_AUTO_PP_DONE = True

    # Report predicted stage weights for debugging.
    cursor = 0
    stage_gib = []
    for rank, count in enumerate(counts):
        b = sum(layer_bytes[cursor:cursor + count])
        if rank == 0:
            b += stage0_extra
        if rank == len(counts) - 1:
            b += last_extra
        stage_gib.append(round(b / (1024 ** 3), 3))
        cursor += count

    print(
        "[TQ PP] auto partition:",
        "pp_size=", pp_size,
        "layers=", n_layers,
        "partition=", os.environ["VLLM_PP_LAYER_PARTITION"],
        "estimated_stage_GiB=", stage_gib,
        "stage0_extra_GiB=", stage0_extra_gib,
        "last_extra_GiB=", last_extra_gib,
        flush=True,
    )

    return counts


# ---------------------------------------------------------------------------
# Qwen4Exp/Qwen3.8 chunked N-gram embedding support
# ---------------------------------------------------------------------------
TQ_NGRAM_CHUNK_ROWS = 1024  # fixed CUDA-kernel N


def _tq_ngram_shard_index(name: str) -> int:
    m = re.search(r"\.shard_(\d+)$", str(name), re.IGNORECASE)
    if m is None:
        raise ValueError(f"Not an N-gram shard record name: {name!r}")
    return int(m.group(1))


def _tq_ngram_group_name(name: str) -> str:
    """Return the manifest group path before `.shard_N`."""
    return re.sub(r"\.shard_\d+$", "", str(name), flags=re.IGNORECASE)


def _tq_resolve_ngram_group(manifest, prefix: str) -> list[str] | None:
    """Resolve one runtime N-gram embedding object to all TQ shard records.

    Qwen3.8/Qwen4Exp checkpoint storage is physically sharded as
    ``...ngram_embedding.shard_N.weight``. Native vLLM, however, constructs one
    giant ``VocabParallelEmbedding`` for the logical ``...ngram_embedding``
    table. The runtime therefore has to bind that ONE module to ALL TQ shard
    records rather than expecting a separate vLLM module per checkpoint shard.
    """
    norm_prefix = _tq_normalize_path(str(prefix))
    low_prefix = norm_prefix.lower()
    if "ngram_embedding" not in low_prefix:
        return None

    groups: dict[str, list[str]] = {}
    for name in manifest.quantized_layers:
        sname = str(name)
        if not re.search(r"\.shard_\d+$", sname, re.IGNORECASE):
            continue
        try:
            meta = manifest.load_meta(sname)
        except Exception:
            continue
        if meta.get("target_kind") != "tq_ngram_chunked_shard":
            continue

        group = str(meta.get("embedding_group") or _tq_ngram_group_name(sname))
        norm_group = _tq_normalize_path(group)
        groups.setdefault(norm_group, []).append(sname)

    if not groups:
        return None

    # Prefer exact normalized group equality.
    matches = [records for group, records in groups.items() if group == norm_prefix]

    # Wrapper differences can still leave one side with extra non-semantic
    # containers. Accept only a UNIQUE suffix-compatible group.
    if not matches:
        compatible = []
        for group, records in groups.items():
            if (
                group.endswith("." + norm_prefix)
                or norm_prefix.endswith("." + group)
                or group == norm_prefix
            ):
                compatible.append(records)
        matches = compatible

    if len(matches) == 0:
        return None
    if len(matches) > 1:
        pretty = [sorted(x, key=_tq_ngram_shard_index)[:3] for x in matches]
        raise RuntimeError(
            f"TQ ambiguous N-gram group resolution for {prefix!r}: {pretty}"
        )

    records = sorted(set(matches[0]), key=_tq_ngram_shard_index)
    if not records:
        return None

    # Validate physical shard ids and global row ranges up front. Gaps in
    # physical IDs are almost certainly a bad/incomplete quantized checkpoint.
    shard_ids = [_tq_ngram_shard_index(x) for x in records]
    expected_ids = list(range(shard_ids[0], shard_ids[-1] + 1))
    if shard_ids != expected_ids:
        raise RuntimeError(
            f"TQ N-gram shard ids are not contiguous for runtime {prefix!r}: "
            f"first={shard_ids[:8]} last={shard_ids[-8:]}"
        )

    return records


class TqNgramEmbeddingMethod(QuantizeMethodBase):
    """One native vLLM embedding backed by many chunked TQ shard records.

    The offline checkpoint remains physically sharded. Each record stores a
    number of independent [M=model-specific,N=1024] TQ matrices. At runtime a global row
    id is mapped to physical shard -> 1024-row chunk -> local column, and the
    selected embedding row is evaluated as TQ(W_chunk) @ one_hot(local_column).
    """

    def __init__(self, quant_config, runtime_prefix: str, records: list[str]):
        self.quant_config = quant_config
        self.runtime_prefix = str(runtime_prefix)
        self.records = list(records)
        if not self.records:
            raise RuntimeError(
                f"TQ N-gram runtime {runtime_prefix!r} resolved zero shard records"
            )

        self.chunk_rows = TQ_NGRAM_CHUNK_ROWS
        self.embedding_dim: int | None = None
        self.shards: list[dict[str, int | str]] = []

        next_row = 0
        total_chunks = 0
        for ordinal, record in enumerate(self.records):
            meta = quant_config.manifest.load_meta(record)
            if meta.get("target_kind") != "tq_ngram_chunked_shard":
                raise RuntimeError(
                    f"TQ N-gram record {record!r} has wrong target_kind: "
                    f"{meta.get('target_kind')!r}"
                )

            saved_chunk_rows = int(meta.get("ngram_chunk_rows") or TQ_NGRAM_CHUNK_ROWS)
            num_chunks = int(meta.get("ngram_num_chunks") or 0)
            valid_rows = int(meta.get("ngram_valid_rows") or 0)

            matrix_shape = meta.get("matrix_shape")
            matrix_m = matrix_n = -1
            if isinstance(matrix_shape, (list, tuple)) and len(matrix_shape) == 2:
                try:
                    matrix_m = int(matrix_shape[0])
                    matrix_n = int(matrix_shape[1])
                except Exception:
                    matrix_m = matrix_n = -1

            embedding_dim_meta = meta.get("ngram_embedding_dim")
            try:
                embedding_dim = int(embedding_dim_meta) if embedding_dim_meta is not None else matrix_m
            except Exception:
                embedding_dim = matrix_m
            if embedding_dim <= 0:
                raise RuntimeError(
                    f"TQ N-gram embedding dimension missing for {record}: "
                    f"ngram_embedding_dim={embedding_dim_meta!r}, matrix_shape={matrix_shape!r}"
                )
            if matrix_m > 0 and matrix_m != embedding_dim:
                raise RuntimeError(
                    f"TQ N-gram M mismatch for {record}: "
                    f"ngram_embedding_dim={embedding_dim}, matrix_shape={matrix_shape}"
                )
            if self.embedding_dim is None:
                self.embedding_dim = embedding_dim
            elif embedding_dim != self.embedding_dim:
                raise RuntimeError(
                    f"TQ N-gram embedding dimension changed within group {self.runtime_prefix!r}: "
                    f"expected M={self.embedding_dim}, got M={embedding_dim} for {record}"
                )

            # N is fixed by the CUDA kernel at 1024. matrix_shape[1] describes
            # the physical N used when this record was quantized. Normal records
            # are N=1024; legacy short tails (N<1024) use the safe zero fallback.
            storage_chunk_rows = matrix_n if matrix_n > 0 else TQ_NGRAM_CHUNK_ROWS

            # The CUDA embedding kernel is fixed at N=1024.  A checkpoint
            # record physically quantized with a shorter N (for example N=32)
            # cannot be losslessly reinterpreted as N=1024 because the FFT/sign
            # transform geometry is different.  For compatibility, mark such
            # records as zero-fill tails and bypass the kernel for those rows.
            # This is intentionally conservative and only affects short tail
            # records; all normal N=1024 records use the compressed TQ kernel.
            zero_fill_tail = storage_chunk_rows != TQ_NGRAM_CHUNK_ROWS
            if zero_fill_tail:
                print(
                    "[TQ NGRAM ZERO TAIL]",
                    "record=", record,
                    "matrix_shape=", matrix_shape,
                    "storage_chunk_rows=", storage_chunk_rows,
                    "kernel_chunk_rows=", TQ_NGRAM_CHUNK_ROWS,
                    "valid_rows=", valid_rows,
                    flush=True,
                )
            row_start = meta.get("embedding_row_start")
            row_end = meta.get("embedding_row_end")
            if row_start is None:
                row_start = next_row
            if row_end is None:
                row_end = int(row_start) + valid_rows
            row_start = int(row_start)
            row_end = int(row_end)

            if saved_chunk_rows != self.chunk_rows:
                print(
                    "[TQ NGRAM TAIL PAD]",
                    "record=", record,
                    "saved_valid_chunk_rows=", saved_chunk_rows,
                    "storage_chunk_rows=", storage_chunk_rows,
                    "valid_rows=", valid_rows,
                    flush=True,
                )
            if num_chunks <= 0 or valid_rows <= 0:
                raise RuntimeError(
                    f"TQ N-gram metadata incomplete for {record}: {meta}"
                )
            if row_end - row_start != valid_rows:
                raise RuntimeError(
                    f"TQ N-gram row-range mismatch for {record}: "
                    f"range=[{row_start},{row_end}) valid_rows={valid_rows}"
                )
            if ordinal and row_start != next_row:
                raise RuntimeError(
                    f"TQ N-gram global row ranges are not contiguous at {record}: "
                    f"expected_start={next_row}, got={row_start}"
                )

            self.shards.append({
                "record": record,
                "shard_id": _tq_ngram_shard_index(record),
                "row_start": row_start,
                "row_end": row_end,
                "valid_rows": valid_rows,
                "num_chunks": num_chunks,
                "storage_chunk_rows": storage_chunk_rows,
                "saved_chunk_rows": saved_chunk_rows,
                "zero_fill_tail": bool(zero_fill_tail),
            })
            next_row = row_end
            total_chunks += num_chunks

        if self.embedding_dim is None or self.embedding_dim <= 0:
            raise RuntimeError(
                f"TQ N-gram runtime {self.runtime_prefix!r} could not derive embedding dimension M"
            )
        self.embedding_dim = int(self.embedding_dim)
        self.valid_rows = int(next_row)
        self.total_chunks = int(total_chunks)
        self.device = _tq_runtime_device()
        self.DL = rand_signs_splitmix_fixed(
            self.chunk_rows, self.device, FIXED_SEED_D_R, out_dtype=torch.float16
        )
        self.DR = rand_signs_splitmix_fixed(
            self.embedding_dim, self.device, FIXED_SEED_D_L, out_dtype=torch.float32
        )[None, :]
        self.use_cufftdx_N = can_use_tq_cufftdx(self.chunk_rows, FFT_SIZES)
        self.use_cufftdx_M = can_use_tq_cufftdx(self.embedding_dim, IFFT_SIZES)
        print(
            "[TQ NGRAM GROUP device]",
            "proc=", multiprocessing.current_process().name,
            "runtime_prefix=", self.runtime_prefix,
            "device=", self.device,
            "physical_shards=", len(self.shards),
            "chunks=", self.total_chunks,
            "rows=", self.valid_rows,
            "M=", self.embedding_dim,
            "N=", self.chunk_rows,
            flush=True,
        )

    @staticmethod
    def _slot_names(ordinal: int) -> tuple[str, str, str]:
        return (
            f"tq_ngram_packed_all_{ordinal}",
            f"tq_ngram_u_{ordinal}",
            f"tq_ngram_std_{ordinal}",
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Register independent compressed storage per physical checkpoint shard.
        # This avoids concatenating/repacking the offline format while still
        # replacing vLLM's one giant dense embedding parameter.
        for ordinal, shard in enumerate(self.shards):
            # Short-N tail records are deliberately zero-filled at lookup time
            # and never enter the fixed-N=1024 CUDA kernel, so they need no
            # compressed runtime parameters.
            if bool(shard.get("zero_fill_tail", False)):
                continue
            num_chunks = int(shard["num_chunks"])
            packed_name, u_name, std_name = self._slot_names(ordinal)
            packed = Parameter(
                torch.empty(
                    (num_chunks, self.embedding_dim, 520),
                    dtype=torch.uint8,
                    device=self.device,
                ),
                requires_grad=False,
            )
            u = Parameter(
                torch.empty((num_chunks,), dtype=torch.float32, device=self.device),
                requires_grad=False,
            )
            std = Parameter(
                torch.empty((num_chunks,), dtype=torch.float32, device=self.device),
                requires_grad=False,
            )
            for param in (packed, u, std):
                set_weight_attrs(param, {"ignore_warning": True})
            layer.register_parameter(packed_name, packed)
            layer.register_parameter(u_name, u)
            layer.register_parameter(std_name, std)

        # Critical: consume the ONE original huge ngram_embedding.weight instead
        # of allowing VocabParallelEmbedding to allocate/load its dense BF16
        # table (~95 GiB in the failing Qwen3.8 run).
        _register_hf_weight_sink(layer, extra_weight_attrs)

    @torch.no_grad()
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        for ordinal, shard in enumerate(self.shards):
            if bool(shard.get("zero_fill_tail", False)):
                continue
            record = str(shard["record"])
            tensors = self.quant_config.manifest.load_tensors(record)
            packed = tensors.get("packed_all")
            if packed is None:
                raise RuntimeError(f"TQ N-gram record {record} has no packed_all")

            num_chunks = int(shard["num_chunks"])
            expected = (num_chunks, self.embedding_dim, 520)
            if tuple(packed.shape) != expected:
                raise RuntimeError(
                    f"TQ N-gram packed shape mismatch for {record}: "
                    f"saved={tuple(packed.shape)} expected={expected}"
                )

            packed_name, u_name, std_name = self._slot_names(ordinal)
            packed_dst = getattr(layer, packed_name)
            u_dst = getattr(layer, u_name)
            std_dst = getattr(layer, std_name)
            packed_dst.copy_(packed.to(device=packed_dst.device, dtype=torch.uint8))
            u_dst.copy_(tensors["u_W"].to(device=u_dst.device, dtype=torch.float32))
            std_dst.copy_(tensors["std_W"].to(device=std_dst.device, dtype=torch.float32))

    def _find_shard_ordinal(self, row: int) -> int:
        # Physical shard count is small (typically ~128). This path is used only
        # once per unique shard touched by a batch; the hot TQ work remains CUDA.
        lo, hi = 0, len(self.shards)
        while lo < hi:
            mid = (lo + hi) // 2
            if row < int(self.shards[mid]["row_end"]):
                hi = mid
            else:
                lo = mid + 1
        if lo >= len(self.shards):
            raise IndexError(row)
        shard = self.shards[lo]
        if row < int(shard["row_start"]):
            raise IndexError(row)
        return lo

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        ids = input_.to(torch.long)
        shape = ids.shape
        flat = ids.reshape(-1)
        if flat.numel() == 0:
            return torch.empty(
                (*shape, self.embedding_dim),
                device=ids.device,
                dtype=torch.float16,
            )
        if bool((flat < 0).any()) or bool((flat >= self.valid_rows).any()):
            raise IndexError(
                f"TQ N-gram row out of range for {self.runtime_prefix}: "
                f"min={int(flat.min())} max={int(flat.max())} "
                f"valid_rows={self.valid_rows}"
            )
        if self.device.type != "cuda":
            raise RuntimeError(
                "TQ N-gram lookup currently uses the CUDA TQ FFT kernel. "
                "Serve the compressed N-gram table on GPU or add a CPU decode kernel."
            )

        out = torch.empty(
            (flat.numel(), self.embedding_dim),
            device=self.device,
            dtype=torch.float16,
        )

        # Determine the touched physical shards on CPU only from the tiny set of
        # unique ids. Avoid copying the full token/id tensor back to CPU.
        unique_rows = torch.unique(flat).detach().cpu().tolist()
        touched_ordinals = sorted({self._find_shard_ordinal(int(r)) for r in unique_rows})

        for ordinal in touched_ordinals:
            shard = self.shards[ordinal]
            row_start = int(shard["row_start"])
            row_end = int(shard["row_end"])
            shard_mask = (flat >= row_start) & (flat < row_end)

            # Compatibility path for a physically short-N checkpoint tail.
            # User explicitly allowed zero filling.  Do not feed the short-N
            # compressed payload to the fixed N=1024 kernel.
            if bool(shard.get("zero_fill_tail", False)):
                shard_positions = torch.nonzero(shard_mask, as_tuple=False).reshape(-1)
                out[shard_positions] = 0
                continue

            shard_rows = flat[shard_mask] - row_start
            chunk_ids = torch.div(shard_rows, self.chunk_rows, rounding_mode="floor")
            local_ids = torch.remainder(shard_rows, self.chunk_rows)

            packed_name, u_name, std_name = self._slot_names(ordinal)
            packed_all = getattr(layer, packed_name)
            u_all = getattr(layer, u_name)
            std_all = getattr(layer, std_name)

            for chunk in torch.unique(chunk_ids).tolist():
                chunk = int(chunk)
                chunk_mask = chunk_ids == chunk
                loc = local_ids[chunk_mask]
                x = torch.zeros(
                    (int(loc.numel()), self.chunk_rows),
                    device=self.device,
                    dtype=torch.float16,
                )
                x.scatter_(1, loc[:, None], 1.0)
                y = _tq_apply_matrix(
                    x,
                    packed_all=packed_all[chunk],
                    u_W=u_all[chunk],
                    std_W=std_all[chunk],
                    DL=self.DL,
                    DR=self.DR,
                    use_cufftdx_N=self.use_cufftdx_N,
                    use_cufftdx_M=self.use_cufftdx_M,
                    N=self.chunk_rows,
                    M=self.embedding_dim,
                    bias=None,
                )

                shard_positions = torch.nonzero(shard_mask, as_tuple=False).reshape(-1)
                out[shard_positions[chunk_mask]] = y

        return out.reshape(*shape, self.embedding_dim)

    # Some vLLM call sites use apply() for embedding-like projection objects.
    def apply(self, layer, input_, bias=None):
        return self.embedding(layer, input_)


@register_quantization_config("tq")
class TqConfig(QuantizationConfig):

    def __init__(
        self,
        manifest: "TqModelManifest | None" = None,
    ) -> None:
        super().__init__()
        # This process is now constructing a TQ model.  Enable the narrow
        # merged-column zero-placeholder compatibility guard before model
        # modules or checkpoint loaders are created.
        _tq_set_runtime_active(True)
        _tq_set_active_config(self)
        # vLLM constructs the quantization config before it supplies the model
        # name/path. Defer manifest resolution until maybe_update_config().
        self.manifest = manifest

        if self.manifest is not None:
            configure_tq_pipeline_partition(self.manifest)

        # No MiMo constructor patch is required here. MTP-vs-backbone
        # ownership is enforced by the manifest namespace resolver.

    def __repr__(self):
        return "TqConfig()"

    @classmethod
    def get_name(cls) -> str:
        return "tq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TqConfig":
        # The model path/repo ID is not part of the quantization config dict.
        # vLLM supplies it immediately afterward through maybe_update_config().
        return cls()

    # def maybe_update_config(
    #     self,
    #     model_name: str,
    #     hf_config=None,
    #     revision: str | None = None,
    # ) -> None:
    def maybe_update_config(
        self,
        model_name,
        hf_config=None,
        revision=None,
    ) -> None:
        """Resolve the TQ manifest from vLLM's normal model argument.

        `model_name` is exactly what the user supplied to vLLM, so both a
        local model directory and a Hugging Face repo ID work. TQ_MODEL_DIR,
        when set, remains an explicit override.
        """
        _tq_set_runtime_active(True)
        _tq_set_active_config(self)
        self.manifest = get_active_tq_manifest(
            model_name,
            revision=revision,
        )

        # Install the memory-aware vLLM PP split before model construction.
        configure_tq_pipeline_partition(self.manifest)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:

        # Re-establish the process-local active config after multiprocessing
        # deserialization before any model-specific embedding constructor runs.
        _tq_set_runtime_active(True)
        _tq_set_active_config(self)
        manifest = self.manifest

        if manifest is None:
            return None

        # ============================================================
        # Qwen4Exp N-gram logical embedding table
        # ============================================================
        # Native vLLM constructs ONE VocabParallelEmbedding for the complete
        # table, while the checkpoint/TQ storage is split into .shard_N records.
        if isinstance(VocabParallelEmbedding, type) and isinstance(layer, VocabParallelEmbedding):
            # Always emit this for now: it proves whether vLLM asks TQ for a
            # quant method for the giant PLE table and exposes the exact prefix.
            print(
                "[TQ EMBEDDING PROBE]",
                "runtime_build=", _TQ_RUNTIME_BUILD,
                "prefix=", repr(prefix),
                "type=", type(layer).__name__,
                "num_embeddings=", getattr(layer, "num_embeddings", None),
                "embedding_dim=", getattr(layer, "embedding_dim", None),
                flush=True,
            )
            ngram_records = _tq_resolve_ngram_group(manifest, prefix)
            if ngram_records is not None:
                print(
                    "[TQ NGRAM GROUP RESOLVED]",
                    "prefix=", repr(prefix),
                    "records=", len(ngram_records),
                    "first=", ngram_records[0],
                    "last=", ngram_records[-1],
                    flush=True,
                )
                return TqNgramEmbeddingMethod(
                    self,
                    runtime_prefix=prefix,
                    records=ngram_records,
                )
            print(
                "[TQ NGRAM GROUP MISS]",
                "prefix=", repr(prefix),
                flush=True,
            )

        # ============================================================
        # Linear
        # ============================================================

        if isinstance(layer, LinearBase):

            manifest_name = manifest.resolve_linear_name(prefix)

            if manifest_name is not None:

                # print(
                #     "[TQ]",
                #     f"runtime={prefix}",
                #     f"manifest={manifest_name}",
                #     "method=TQ",
                #     flush=True,
                # )

                return TqLinearMethod(
                    layer.input_size,
                    layer.output_size,
                    quant_config=self,
                    prefix=manifest_name,
                )

            # --------------------------------------------------------
            # Legacy packed checkpoint support
            # --------------------------------------------------------

            packed = manifest.resolve_packed_linear(prefix)

            if packed is not None:
                packed_kind, source_names = packed

                if packed_kind == "qkv":
                    return TqQKVMethod(
                        layer.input_size,
                        layer.output_size,
                        quant_config=self,
                        prefixes=source_names,
                    )

                if packed_kind == "gate_up":
                    return TqGateUpMethod(
                        layer.input_size,
                        layer.output_size,
                        quant_config=self,
                        prefixes=source_names,
                    )

                if packed_kind in ("qkvz", "ba"):
                    return TqConcatPackedMethod(
                        layer.input_size,
                        layer.output_size,
                        quant_config=self,
                        prefixes=source_names,
                        kind=packed_kind,
                    )

                raise RuntimeError(
                    "TQ: unsupported packed linear kind "
                    f"{packed_kind!r} for prefix={prefix!r}"
                )

            # A genuinely unquantized layer is allowed.  Emit one concise
            # diagnostic when requested so residual-checkpoint mistakes are
            # obvious instead of surfacing later as an opaque narrow(..., 0)
            # failure in vLLM.
            if os.environ.get("TQ_DEBUG_DISPATCH", "0") == "1":
                known_native_packed = any(
                    prefix.endswith("." + suffix) or prefix == suffix
                    for suffix in (
                        "qkv_proj", "gate_up_proj",
                        "in_proj_qkvz", "in_proj_ba",
                    )
                )
                print(
                    "[TQ LINEAR UNQUANTIZED]",
                    "prefix=", prefix,
                    "type=", type(layer).__name__,
                    "input=", getattr(layer, "input_size", None),
                    "output=", getattr(layer, "output_size", None),
                    "known_native_packed=", known_native_packed,
                    flush=True,
                )
                if known_native_packed:
                    print(
                        "[TQ PACKED MANIFEST MISS]",
                        "runtime=", prefix,
                        "normalized=", _tq_normalize_path(prefix),
                        "hint=checkpoint was likely quantized without the "
                        "native packed target; re-quantize with the fixed quantizer",
                        flush=True,
                    )
            return UnquantizedLinearMethod()

        # ============================================================
        # MoE
        # ============================================================

        # if (
        #     isinstance(RoutedExperts, type)
        #     and isinstance(layer, RoutedExperts)
        # ):

        #     manifest_name = manifest.resolve_runtime_name(prefix)

        #     if manifest_name is None:
        #         return None

        #     resolved = manifest.resolve_expert_group(manifest_name)

        #     if resolved is None:
        #         return None

        #     return TqMoEMethod(
        #         moe_config=layer.moe_config,
        #         quant_config=self,
        #         prefix=manifest_name,
        #         manifest=manifest,
        #     )
        if (
            isinstance(RoutedExperts, type)
            and isinstance(layer, RoutedExperts)
        ):
            print(
                "[TQ MOE dispatch]",
                "runtime prefix=", repr(prefix),
                "layer type=", type(layer),
                flush=True,
            )

            resolved = manifest.resolve_expert_group(prefix)

            print(
                "[TQ MOE dispatch]",
                "resolved=", resolved[0] if resolved else None,
                flush=True,
            )

            if resolved is None:
                return None

            group_name, _ = resolved

            print(
                "[TQ MOE dispatch]",
                "using TqMoEMethod group=", group_name,
                flush=True,
            )

            return TqMoEMethod(
                moe_config=layer.moe_config,
                quant_config=self,
                prefix=group_name,
                manifest=manifest,
            )


        return None

    







def _tq_apply_matrix(
    input: torch.Tensor,
    *,
    packed_all: torch.Tensor,
    u_W: torch.Tensor,
    std_W: torch.Tensor,
    DL: torch.Tensor,
    DR: torch.Tensor,
    use_cufftdx_N: bool,
    use_cufftdx_M: bool,
    N: int,
    M: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one TQ-quantized matrix.

    This is the common implementation used by normal Linear layers and by
    individual MoE expert gate_up/down matrices.

    Logical weight shape is [M, N], input shape is [..., N].
    """
    in_dtype = input.dtype
    in_shape = input.shape

    # if N % 2 != 0 or M % 2 != 0:
    #     raise ValueError(
    #         f"TQ FFT path requires even N and M, got N={N}, M={M}"
    #     )
    x_in = input.reshape(-1, input.shape[-1]).to(torch.float16)

    x, input_sum = torch.ops.my_qlinear.forward_fft(x_in, N, DL, use_cufftdx_N)


    x = x.contiguous()

    # # forward_pass is expected to dispatch internally between your GEMV,
    # # small-L MMA, and decompress+GEMM paths.

    # if packed_all.dtype != torch.uint8:
    #     raise RuntimeError(
    #         f"TQ bad packed_all before forward_pass: "
    #         f"dtype={packed_all.dtype}, "
    #         f"shape={tuple(packed_all.shape)}, "
    #         f"N={N}, M={M}"
    #     )
    x = torch.ops.my_qlinear.forward_pass(
        packed_all,
        x,
        N,
        M,
    )


    # ============================================================
    # Inverse FFT + epilogue
    # ============================================================
    x = torch.ops.my_qlinear.reverse_ifft(
        x,
        DR,
        std_W,
        input_sum,
        u_W,
        bias,
        use_cufftdx_M
    )

    out = x.reshape(
        *in_shape[:-1],
        M,
    )

    return out.to(in_dtype)




TQ_MATRIX_TENSOR_KEYS = (
    "SigRec1_select_packed",
    "SigRec2_select_packed",
    "SigRec3_select_packed",
    "SigRec4_select_packed",
    "X567_packed",
    "u_W",
    "std_W",
)





def _tq_record_matrix_shape(manifest: TqModelManifest, layer_name: str) -> tuple[int, int]:
    meta = manifest.load_meta(layer_name)
    shape = meta.get("matrix_shape") or meta.get("original_shape")
    if shape is None or len(shape) != 2:
        raise RuntimeError(
            f"TQ record {layer_name!r} has invalid/missing matrix shape: {shape!r}"
        )
    return int(shape[0]), int(shape[1])


def _assert_tq_tp1(
    *,
    input_size_per_partition: int,
    output_partition_sizes: list[int],
    input_size: int,
    output_size: int,
    prefix: str,
) -> None:
    """Native packed TQ currently stores full matrices, so require TP=1."""
    local_out = sum(int(x) for x in output_partition_sizes)
    if int(input_size_per_partition) != int(input_size) or local_out != int(output_size):
        raise RuntimeError(
            "TQ native-vLLM packed layers currently require tensor_parallel_size=1. "
            f"prefix={prefix!r}, input_size_per_partition={input_size_per_partition}, "
            f"input_size={input_size}, output_partition_sizes={output_partition_sizes}, "
            f"output_size={output_size}"
        )


def _register_tq_matrix(
    layer: torch.nn.Module,
    slot: str,
    N: int,
    M: int,
) -> None:
    """Register one full TQ matrix using the final [B, 520] runtime layout."""
    B = (int(N) * int(M) + 1023) // 1024

    device = _tq_runtime_device()

    packed_all = Parameter(
        torch.zeros((B, 520), device=device, dtype=torch.uint8),
        requires_grad=False,
    )
    u_W = Parameter(
        torch.empty((), device=device, dtype=torch.float32),
        requires_grad=False,
    )
    std_W = Parameter(
        torch.empty((), device=device, dtype=torch.float32),
        requires_grad=False,
    )

    for param in (packed_all, u_W, std_W):
        set_weight_attrs(param, {"ignore_warning": True})

    layer.register_parameter(f"tq_{slot}_packed_all", packed_all)
    layer.register_parameter(f"tq_{slot}_u", u_W)
    layer.register_parameter(f"tq_{slot}_std", std_W)


@torch.no_grad()
def _copy_tq_record_into_named_slot(
    layer: torch.nn.Module,
    manifest: TqModelManifest,
    prefix: str,
    slot: str,
) -> None:
    tensors = manifest.load_tensors(prefix)
    dst = getattr(layer, f"tq_{slot}_packed_all")
    u_dst = getattr(layer, f"tq_{slot}_u")
    std_dst = getattr(layer, f"tq_{slot}_std")
    B = dst.shape[0]

    # Preferred safetensors format already stores the final packed layout.
    if "packed_all" in tensors:
        packed = tensors["packed_all"]
        if tuple(packed.shape) != tuple(dst.shape):
            raise RuntimeError(
                f"TQ packed_all shape mismatch for {prefix}: "
                f"saved={tuple(packed.shape)} runtime={tuple(dst.shape)}"
            )
        dst.copy_(packed.to(device=dst.device, dtype=torch.uint8))
    else:
        # Legacy per-layer/.pt representation.
        sig1 = tensors["SigRec1_select_packed"]
        sig2 = tensors["SigRec2_select_packed"]
        sig3 = tensors["SigRec3_select_packed"]
        sig4 = tensors["SigRec4_select_packed"]
        x567 = tensors["X567_packed"]

        if tuple(sig1.shape) != (B, 7):
            raise RuntimeError(f"TQ sig1 shape mismatch for {prefix}: {tuple(sig1.shape)}")
        if tuple(sig2.shape) != (B, 12):
            raise RuntimeError(f"TQ sig2 shape mismatch for {prefix}: {tuple(sig2.shape)}")
        if tuple(sig3.shape) != (B, 29):
            raise RuntimeError(f"TQ sig3 shape mismatch for {prefix}: {tuple(sig3.shape)}")
        if tuple(sig4.shape) != (B, 83):
            raise RuntimeError(f"TQ sig4 shape mismatch for {prefix}: {tuple(sig4.shape)}")
        if tuple(x567.shape) != (3 * B, 128):
            raise RuntimeError(f"TQ X567 shape mismatch for {prefix}: {tuple(x567.shape)}")

        dst[:, 0:7].copy_(sig1.to(device=dst.device, dtype=torch.uint8))
        dst[:, 8:20].copy_(sig2.to(device=dst.device, dtype=torch.uint8))
        dst[:, 20:49].copy_(sig3.to(device=dst.device, dtype=torch.uint8))
        dst[:, 52:135].copy_(sig4.to(device=dst.device, dtype=torch.uint8))
        dst[:, 136:264].copy_(x567[:B].to(device=dst.device, dtype=torch.uint8))
        dst[:, 264:392].copy_(x567[B:2 * B].to(device=dst.device, dtype=torch.uint8))
        dst[:, 392:520].copy_(x567[2 * B:3 * B].to(device=dst.device, dtype=torch.uint8))

    u_dst.copy_(tensors["u_W"].to(device=dst.device, dtype=torch.float32).reshape(()))
    std_dst.copy_(tensors["std_W"].to(device=dst.device, dtype=torch.float32).reshape(()))


class _TqPackedLinearMethodBase(LinearMethodBase):
    """Shared utilities for native-vLLM packed linear adapters."""

    def _init_runtime_for_slot(self, slot: str, N: int, M: int) -> None:
        DL = rand_signs_splitmix_fixed(
            N, "cuda", FIXED_SEED_D_R, out_dtype=torch.float16
        )
        DR = rand_signs_splitmix_fixed(
            M, "cuda", FIXED_SEED_D_L, out_dtype=torch.float32
        )[None, :]
        setattr(self, f"_{slot}_DL", DL)
        setattr(self, f"_{slot}_DR", DR)
        setattr(self, f"_{slot}_use_cufftdx_N", can_use_tq_cufftdx(N, FFT_SIZES))
        setattr(self, f"_{slot}_use_cufftdx_M", can_use_tq_cufftdx(M, IFFT_SIZES))

    def _apply_slot(
        self,
        layer: torch.nn.Module,
        input: torch.Tensor,
        slot: str,
        N: int,
        M: int,
    ) -> torch.Tensor:
        return _tq_apply_matrix(
            input,
            packed_all=getattr(layer, f"tq_{slot}_packed_all"),
            u_W=getattr(layer, f"tq_{slot}_u"),
            std_W=getattr(layer, f"tq_{slot}_std"),
            DL=getattr(self, f"_{slot}_DL"),
            DR=getattr(self, f"_{slot}_DR"),
            use_cufftdx_N=getattr(self, f"_{slot}_use_cufftdx_N"),
            use_cufftdx_M=getattr(self, f"_{slot}_use_cufftdx_M"),
            N=N,
            M=M,
            bias=None,
        )


class TqQKVMethod(_TqPackedLinearMethodBase):
    """Adapter for native vLLM QKVParallelLinear using 3 HF TQ records."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        quant_config: TqConfig,
        prefixes: list[str],
    ):
        if len(prefixes) != 3:
            raise ValueError(f"TqQKVMethod expected 3 prefixes, got {prefixes!r}")

        self.quant_config = quant_config
        self.prefixes = list(prefixes)
        self.q_prefix, self.k_prefix, self.v_prefix = self.prefixes
        self.N = ((int(input_size) + 31) // 32) * 32
        self.output_size = int(output_size)

        q_M, q_N = _tq_record_matrix_shape(quant_config.manifest, self.q_prefix)
        k_M, k_N = _tq_record_matrix_shape(quant_config.manifest, self.k_prefix)
        v_M, v_N = _tq_record_matrix_shape(quant_config.manifest, self.v_prefix)
        if not (q_N == k_N == v_N == self.N):
            raise RuntimeError(
                f"TQ QKV input-size mismatch: runtime N={self.N}, "
                f"q={q_M, q_N}, k={k_M, k_N}, v={v_M, v_N}"
            )
        if q_M + k_M + v_M != self.output_size:
            raise RuntimeError(
                f"TQ QKV output-size mismatch: q+k+v={q_M+k_M+v_M}, "
                f"runtime output_size={self.output_size}"
            )

        self.q_M, self.k_M, self.v_M = q_M, k_M, v_M
        self._init_runtime_for_slot("q", self.N, self.q_M)
        self._init_runtime_for_slot("k", self.N, self.k_M)
        self._init_runtime_for_slot("v", self.N, self.v_M)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        _assert_tq_tp1(
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            prefix=f"QKV({self.q_prefix}, {self.k_prefix}, {self.v_prefix})",
        )
        _register_tq_matrix(layer, "q", self.N, self.q_M)
        _register_tq_matrix(layer, "k", self.N, self.k_M)
        _register_tq_matrix(layer, "v", self.N, self.v_M)

        # Crucial for native vLLM: HF q/k/v checkpoint tensors are mapped into
        # this one qkv_proj parameter with shard_id='q'/'k'/'v'. The sink accepts
        # those calls (including shard_id) and prevents normal merged-weight
        # slicing of TQ-owned zero-sized residual placeholders.
        _register_hf_weight_sink(layer, extra_weight_attrs)

    @torch.no_grad()
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _copy_tq_record_into_named_slot(layer, self.quant_config.manifest, self.q_prefix, "q")
        _copy_tq_record_into_named_slot(layer, self.quant_config.manifest, self.k_prefix, "k")
        _copy_tq_record_into_named_slot(layer, self.quant_config.manifest, self.v_prefix, "v")

    def apply(
        self,
        layer: torch.nn.Module,
        input: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._apply_slot(layer, input, "q", self.N, self.q_M)
        k = self._apply_slot(layer, input, "k", self.N, self.k_M)
        v = self._apply_slot(layer, input, "v", self.N, self.v_M)
        out = torch.cat((q, k, v), dim=-1)
        if bias is not None:
            out = out + bias
        return out


class TqConcatPackedMethod(_TqPackedLinearMethodBase):
    """Legacy adapter for a native vLLM packed linear backed by split TQ records.

    Qwen GDN uses this for:
      * in_proj_qkvz <- in_proj_qkv + in_proj_z
      * in_proj_ba   <- in_proj_b + in_proj_a

    Newer quantizer builds emit one fused TQ record under the native runtime
    name, in which case get_quant_method selects TqLinearMethod directly.
    This class keeps older/split manifests loadable.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        quant_config: TqConfig,
        prefixes: list[str],
        kind: str,
    ):
        if len(prefixes) < 2:
            raise ValueError(
                f"TqConcatPackedMethod expected >=2 prefixes, got {prefixes!r}"
            )
        self.quant_config = quant_config
        self.prefixes = list(prefixes)
        self.kind = str(kind)
        self.N = ((int(input_size) + 31) // 32) * 32
        self.output_size = int(output_size)
        self._slots: list[tuple[str, str, int]] = []

        total_M = 0
        for i, record_prefix in enumerate(self.prefixes):
            M, N = _tq_record_matrix_shape(quant_config.manifest, record_prefix)
            if int(N) != int(self.N):
                raise RuntimeError(
                    f"TQ {self.kind} input-size mismatch for {record_prefix!r}: "
                    f"saved={(M, N)}, runtime N={self.N}"
                )
            slot = f"p{i}"
            self._slots.append((slot, record_prefix, int(M)))
            self._init_runtime_for_slot(slot, self.N, int(M))
            total_M += int(M)

        if total_M != self.output_size:
            raise RuntimeError(
                f"TQ {self.kind} output-size mismatch: sources total={total_M}, "
                f"runtime output_size={self.output_size}, prefixes={self.prefixes!r}"
            )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        _assert_tq_tp1(
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            prefix=f"{self.kind}({', '.join(self.prefixes)})",
        )
        for slot, _record_prefix, M in self._slots:
            _register_tq_matrix(layer, slot, self.N, M)
        _register_hf_weight_sink(layer, extra_weight_attrs)

    @torch.no_grad()
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        for slot, record_prefix, _M in self._slots:
            _copy_tq_record_into_named_slot(
                layer, self.quant_config.manifest, record_prefix, slot
            )

    def apply(
        self,
        layer: torch.nn.Module,
        input: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [
            self._apply_slot(layer, input, slot, self.N, M)
            for slot, _record_prefix, M in self._slots
        ]
        out = torch.cat(parts, dim=-1)
        if bias is not None:
            out = out + bias
        return out


class TqGateUpMethod(_TqPackedLinearMethodBase):
    """Adapter for native vLLM MergedColumnParallelLinear gate_up_proj."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        quant_config: TqConfig,
        prefixes: list[str],
    ):
        if len(prefixes) != 2:
            raise ValueError(f"TqGateUpMethod expected 2 prefixes, got {prefixes!r}")

        self.quant_config = quant_config
        self.prefixes = list(prefixes)
        self.gate_prefix, self.up_prefix = self.prefixes
        self.N = ((int(input_size) + 31) // 32) * 32
        self.output_size = int(output_size)

        gate_M, gate_N = _tq_record_matrix_shape(quant_config.manifest, self.gate_prefix)
        up_M, up_N = _tq_record_matrix_shape(quant_config.manifest, self.up_prefix)
        if gate_N != self.N or up_N != self.N:
            raise RuntimeError(
                f"TQ gate/up input-size mismatch: runtime N={self.N}, "
                f"gate={gate_M, gate_N}, up={up_M, up_N}"
            )
        if gate_M + up_M != self.output_size:
            raise RuntimeError(
                f"TQ gate/up output-size mismatch: gate+up={gate_M+up_M}, "
                f"runtime output_size={self.output_size}"
            )

        self.gate_M, self.up_M = gate_M, up_M
        self._init_runtime_for_slot("gate", self.N, self.gate_M)
        self._init_runtime_for_slot("up", self.N, self.up_M)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        _assert_tq_tp1(
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            prefix=f"gate_up({self.gate_prefix}, {self.up_prefix})",
        )
        _register_tq_matrix(layer, "gate", self.N, self.gate_M)
        _register_tq_matrix(layer, "up", self.N, self.up_M)
        _register_hf_weight_sink(layer, extra_weight_attrs)

    @torch.no_grad()
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _copy_tq_record_into_named_slot(
            layer, self.quant_config.manifest, self.gate_prefix, "gate"
        )
        _copy_tq_record_into_named_slot(
            layer, self.quant_config.manifest, self.up_prefix, "up"
        )

    def apply(
        self,
        layer: torch.nn.Module,
        input: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gate = self._apply_slot(layer, input, "gate", self.N, self.gate_M)
        up = self._apply_slot(layer, input, "up", self.N, self.up_M)
        out = torch.cat((gate, up), dim=-1)
        if bias is not None:
            out = out + bias
        return out


class TqLinearMethod(LinearMethodBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        quant_config: TqConfig,
        prefix: str,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.quant_config = quant_config
        self.prefix = prefix

        saved_M, saved_N = _tq_record_matrix_shape(
            quant_config.manifest,
            prefix,
        )

        hf_config = quant_config.manifest.hf_config
        self._mimo_qkv_spec = _tq_mimo_qkv_layout_spec(
            hf_config,
            prefix,
            saved_M,
        )

        # Input dimension is logical/pre-padding metadata and must match.
        if saved_N != input_size:
            raise RuntimeError(
                f"TQ/native-vLLM input shape mismatch for {prefix!r}: "
                f"saved N={saved_N}, runtime N={input_size}"
            )

        # Normally output dimensions must match exactly. For MiMo MTP, the
        # companion vLLM construction patch above should make runtime M == saved M
        # (14848 for SWA). Keep this strict check so a failed/changed upstream
        # patch cannot silently run a malformed model.
        if saved_M != output_size:
            raise RuntimeError(
                f"TQ/native-vLLM output shape mismatch for {prefix!r}: "
                f"saved M={saved_M}, runtime M={output_size}. "
                f"MiMo QKV spec={self._mimo_qkv_spec!r}"
            )


        # self.num_batches_per_block=1<<16

        self.k_polar = 10
        self.N_polar = 1<<self.k_polar
        # self.ReverseIndex = bit_reverse_indices(self.N_polar, device="cuda")
        # self.SCLayer = polar_sc_decode_prepare(self.k_polar, device="cuda")
        self.factor = 0.0988 #0.5/5.0596

        self.ISNum1_n10 = 56
        self.ISNum2_n10 = 96
        self.ISNum3_n10 = 232
        self.ISNum4_n10 = 664
        self.ISNum1_n10_bytes = 7
        self.ISNum2_n10_bytes = 12
        self.ISNum3_n10_bytes = 29
        self.ISNum4_n10_bytes = 83
        self.device = _tq_runtime_device()


        self.N_polar_bytes = self.N_polar>>3
        N, M = input_size, output_size
        self.N=((int(N) + 31) // 32) * 32

        self.DL=rand_signs_splitmix_fixed(self.N, self.device, FIXED_SEED_D_R, out_dtype=torch.float16)
        DR=rand_signs_splitmix_fixed(output_size, self.device, FIXED_SEED_D_L, out_dtype=torch.float32)
        self.DR = DR[None,:]

        print(
            "[TQ LINEAR device]",
            "proc=", multiprocessing.current_process().name,
            "prefix=", self.prefix,
            "device=", self.device,
            flush=True,
        )

        self.use_cufftdx_N = can_use_tq_cufftdx(self.N, FFT_SIZES)
        self.use_cufftdx_M = can_use_tq_cufftdx(M, IFFT_SIZES)

        self.M=int(M)
        self.max_nm = max(self.N, self.M)
        self.N_cplx = int(self.N/2)
        self.M_cplx = int(M/2)
        self.is_pow2_n = (self.N_cplx & (self.N_cplx - 1)) == 0
        self.is_pow2_m = (self.M_cplx & (self.M_cplx - 1)) == 0
        T = M*self.N
        batch_size = self.N_polar
        self.num_batches = (T + batch_size - 1)//batch_size



    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        _assert_tq_tp1(
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            prefix=self.prefix,
        )
        B = self.num_batches

        packed_all = Parameter(
            torch.zeros(
                (
                    B,
                    8 + 12 + 32 + 84 + 128 * 3,
                ),
                device=self.device,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )

        set_weight_attrs(
            packed_all,
            {"ignore_warning": True},
        )

        layer.register_parameter(
            "packed_all",
            packed_all,
        )

        u_W = Parameter(
            torch.empty(
                (),
                device=self.device,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(u_W, {"ignore_warning": True})
        layer.register_parameter("u_W", u_W)

        std_W = Parameter(
            torch.empty(
                (),
                device=self.device,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(std_W, {"ignore_warning": True})
        layer.register_parameter("std_W", std_W)

        _register_hf_weight_sink(layer, extra_weight_attrs)

    @torch.no_grad()
    def process_weights_after_loading(
        self,
        layer: torch.nn.Module,
    ) -> None:

        _load_tq_record_into_packed(
            layer,
            self.quant_config.manifest,
            self.prefix,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        input: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # print(
        #     "[TQ APPLY]",
        #     "prefix=", self.prefix,
        #     "input_shape=", tuple(input.shape),
        #     "input_numel=", input.numel(),
        #     "layer_input_size=", layer.input_size,
        #     "layer_output_size=", layer.output_size,
        #     "tq_N=", self.N,
        #     "tq_M=", self.M,
        #     flush=True,
        # )
        
        out = _tq_apply_matrix(
            input,
            packed_all = layer.packed_all,
            u_W=layer.u_W,#.to(torch.float16),
            std_W=layer.std_W,#.to(torch.float16),
            DL=self.DL,
            DR=self.DR,
            use_cufftdx_N = self.use_cufftdx_N,
            use_cufftdx_M = self.use_cufftdx_M,
            N=self.N,
            M=self.M,
            bias=bias,
        )

        # Existing MiMo TQ checkpoints were quantized from the checkpoint's
        # grouped fused-QKV rows. Reorder the resulting activations into the
        # contiguous Q/K/V layout expected by vLLM QKVParallelLinear.
        if self._mimo_qkv_spec is not None:
            out = _tq_mimo_deinterleave_qkv_output(
                out,
                self._mimo_qkv_spec,
            )

        return out


class TqMoEMethod(FusedMoEMethodBase):
    """Manifest-driven, correctness-first TQ routed-expert implementation.

    Supports the expert layouts present in the attached checkpoints:
      * gate_proj / up_proj / down_proj
      * w1 / w3 / w2
      * fused gate_up_proj
      * projection-before-expert (`down_proj.expert_0007`)
      * arbitrary per-expert partial TQ coverage

    TQ expert matrices are stored expert-major so RoutedExperts.get_expert_weights
    can safely view every Parameter as [local_num_experts, -1]. Dense fallbacks are
    non-persistent buffers and are allocated ONLY for missing expert matrices.

    Current compressed-matrix math is full-matrix and therefore TP-sharding a
    quantized expert is not supported. Use TP=1 for these experts and scale with
    expert parallelism; a future packed-shard format can relax this restriction.
    """

    _SIG_LOGICAL = {"sig1": 7, "sig2": 12, "sig3": 29, "sig4": 83}
    _SIG_PADDED = {"sig1": 8, "sig2": 12, "sig3": 32, "sig4": 84}

    def __init__(self, moe_config, quant_config: TqConfig, prefix: str, manifest: TqModelManifest):
        super().__init__(moe_config)
        self.quant_config = quant_config
        self.prefix = prefix
        self.manifest = manifest
        self.moe_config = moe_config
        self.moe = moe_config  # optional compatibility alias
        self.N_polar = 1024
        self.num_experts = 0                 # local physical experts
        self.global_num_experts = 0
        self.hidden_size = 0
        self.intermediate_size = 0           # full I (TP=1 for TQ)
        self.params_dtype = torch.float16
        self.DL_gu = self.DR_gu = None
        self.DL_down = self.DR_down = None

        self.group_name: str | None = None
        self._records: dict[int, dict[str, str]] = {}
        self._record_raw_projection: dict[tuple[int, str], str] = {}
        self._local_to_global: list[int] = []
        self._global_to_local: dict[int, int] = {}
        self._registered_roles: set[str] = set()
        self._role_signs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        # role -> local expert id -> registered per-expert buffer name.
        # Keeping each missing matrix independent avoids allocating a giant
        # [num_missing_experts, ...] contiguous fallback bank.
        self._dense_slots: dict[str, dict[int, str]] = {
            r: {} for r in ("gate", "up", "down")
        }

    @property
    def supports_eplb(self) -> bool:
        return False

    @property
    def allow_inplace(self) -> bool:
        return False

    def get_fused_moe_quant_config(self, layer):
        return None

    def _resolve_local_global_ids(self, layer, E: int, global_E: int) -> list[int]:
        table = getattr(layer, "expert_local_to_global", None)
        if isinstance(table, torch.Tensor) and table.numel() >= E:
            vals = table.detach().cpu().reshape(-1).tolist()[:E]
            return [int(x) for x in vals]
        # TP=1 / EP=1 and older vLLM.
        if E == global_E:
            return list(range(E))
        # Some older EP layouts expose expert_map (global -> local).
        emap = getattr(layer, "expert_map", None)
        if isinstance(emap, torch.Tensor):
            cpu = emap.detach().cpu().reshape(-1)
            pairs = [(int(local), int(g)) for g, local in enumerate(cpu.tolist()) if int(local) >= 0]
            pairs.sort()
            if len(pairs) >= E:
                return [g for _, g in pairs[:E]]
        raise RuntimeError(
            f"TQ MoE {self.prefix}: cannot determine local->global expert IDs "
            f"(local={E}, global={global_E})."
        )

    def _register_tq_role(
        self,
        layer,
        role: str,
        B: int,
        E: int,
    ) -> None:
        pfx = f"tq_{role}"

        # Final runtime layout:
        #
        # sig1 :   8
        # sig2 :  12
        # sig3 :  32
        # sig4 :  84
        # X5   : 128
        # X6   : 128
        # X7   : 128
        #
        # total = 520 bytes / polar block
        # packed_all = Parameter(
        #     torch.zeros(
        #         (E, B, 520),
        #         device="cuda",
        #         dtype=torch.uint8,
        #     ),
        #     requires_grad=False,
        # )

        device = getattr(self, "_tq_device", _tq_runtime_device())

        packed_all = Parameter(
            torch.zeros(
                (E, B, 520),
                device=device,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )


        set_weight_attrs(
            packed_all,
            {"ignore_warning": True},
        )

        layer.register_parameter(
            f"{pfx}_packed_all",
            packed_all,
        )

        # u = Parameter(
        #     torch.empty(
        #         (E,),
        #         device="cuda",
        #         dtype=torch.float32,
        #     ),
        #     requires_grad=False,
        # )

        u = Parameter(
            torch.empty(
                (E,),
                device=device,
                dtype=torch.float32,
            ),
                requires_grad=False,
        )



        # std = Parameter(
        #     torch.empty(
        #         (E,),
        #         device="cuda",
        #         dtype=torch.float32,
        #     ),
        #     requires_grad=False,
        # )

        std = Parameter(
            torch.empty(
                (E,),
                device=device,
                dtype=torch.float32,
            ),
                requires_grad=False,
        )

        set_weight_attrs(u, {"ignore_warning": True})
        set_weight_attrs(std, {"ignore_warning": True})

        layer.register_parameter(
            f"{pfx}_u",
            u,
        )

        layer.register_parameter(
            f"{pfx}_std",
            std,
        )

        self._registered_roles.add(role)

    def _record_matrix_shape(self, layer_name: str) -> tuple[int, int]:
        meta = self.manifest.load_meta(layer_name)
        shape = meta.get("matrix_shape") or meta.get("original_shape")
        if shape is None:
            raise RuntimeError(f"TQ record {layer_name!r} has no matrix shape metadata")
        return tuple(map(int, shape))

    def _build_record_plan(self, H: int, I: int) -> None:
        refs = self.manifest.resolve_expert_group(self.prefix)
        if not refs:
            raise RuntimeError(f"TQ: no expert records resolve to RoutedExperts prefix {self.prefix!r}")
        self.group_name, refs_by_expert = refs
        records: dict[int, dict[str, str]] = {}
        raw_names: dict[tuple[int, str], str] = {}
        role_example: dict[str, str] = {}

        for gid, refs_for_expert in refs_by_expert.items():
            for ref in refs_for_expert:
                shape = self._record_matrix_shape(ref.layer_name)
                role = _tq_record_role(ref.raw_projection, shape, H, I)
                if role in records.setdefault(gid, {}):
                    old = records[gid][role]
                    raise RuntimeError(
                        f"TQ {self.prefix}: two records map to role {role!r} for expert {gid}: "
                        f"{old!r} and {ref.layer_name!r}"
                    )
                records[gid][role] = ref.layer_name
                raw_names[(gid, role)] = ref.raw_projection
                role_example.setdefault(role, ref.layer_name)

        self._records = records
        self._record_raw_projection = raw_names

        # Register only roles that actually occur. B is determined from the saved
        # record shape rather than assuming all models use HxI exactly.
        for role, example in role_example.items():
            out_f, in_f = self._record_matrix_shape(example)
            N_pad = ((in_f + 31) // 32) * 32
            T = out_f * N_pad
            B = (T + self.N_polar - 1) // self.N_polar
            self._role_dims[role] = (N_pad, out_f)
            self._role_batches[role] = B

    def _component_is_quantized(self, gid: int, component: str) -> bool:
        rec = self._records.get(gid, {})
        if component in ("gate", "up") and "gate_up" in rec:
            return True
        return component in rec

    def _allocate_dense_fallbacks(
        self,
        layer,
        H: int,
        I: int,
        dtype: torch.dtype,
    ) -> None:
        """Allocate dense fallback storage only for exact missing expert/role pairs.

        Previous code grouped all missing experts for one role into one tensor:
            [num_missing, out_features, in_features]

        For MiMo a completely missing projection across 256 experts can be about
        4 GiB in FP16, so CUDA had to satisfy one enormous contiguous allocation.

        This implementation registers one buffer per exact missing
        (local_expert_id, role) pair. It does NOT allocate any fallback matrix
        for a quantized component.

        Note: if every expert genuinely lacks a role, the total memory required
        is unchanged; it is simply no longer one giant contiguous allocation.
        """
        needs_gate = bool(getattr(self.moe_config, "is_act_and_mul", True))
        components = ["up", "down"] + (["gate"] if needs_gate else [])

        elem_size = torch.empty((), dtype=dtype).element_size()
        device = getattr(
            self,
            "_tq_device",
            _tq_runtime_device(),
        )
        for role in components:
            missing_local = [
                lid
                for lid, gid in enumerate(self._local_to_global)
                if not self._component_is_quantized(gid, role)
            ]

            role_map: dict[int, str] = {}
            self._dense_slots[role] = role_map

            if role in ("gate", "up"):
                matrix_shape = (I, H)
            else:
                matrix_shape = (H, I)

            bytes_per_matrix = (
                int(matrix_shape[0])
                * int(matrix_shape[1])
                * int(elem_size)
            )

            for lid in missing_local:
                buffer_name = f"tq_dense_{role}_expert_{lid}"

                # layer.register_buffer(
                #     buffer_name,
                #     torch.empty(
                #         matrix_shape,
                #         device="cuda",
                #         dtype=dtype,
                #     ),
                #     persistent=False,
                # )

                layer.register_buffer(
                    buffer_name,
                    torch.empty(
                        matrix_shape,
                        device=device,
                        dtype=dtype,
                    ),
                    persistent=False,
                )



                role_map[lid] = buffer_name

            total_bytes = bytes_per_matrix * len(missing_local)
            print(
                "[TQ MOE sparse fallback]",
                "prefix=", self.prefix,
                "role=", role,
                "missing_local=", len(missing_local),
                "local_experts=", self.num_experts,
                "bytes_per_expert=", bytes_per_matrix,
                "total_GiB=", round(total_bytes / (1024 ** 3), 4),
                "global_ids=", [
                    int(self._local_to_global[lid])
                    for lid in missing_local[:16]
                ],
                ("..." if len(missing_local) > 16 else ""),
                flush=True,
            )

    def create_weights(
        self,
        layer,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        proc_name = multiprocessing.current_process().name

        self._tq_device = torch.device(
            "cpu" if _tq_is_ple_offload_worker() else "cuda"
        )

        print(
            "[TQ MOE device]",
            "proc=", proc_name,
            "prefix=", self.prefix,
            "device=", self._tq_device,
            flush=True,
        )
        
        E = int(num_experts)
        H = int(hidden_size)
        H = ((H + 31) // 32) * 32
        global_E = int(extra_weight_attrs.get("global_num_experts", getattr(self.moe_config, "num_experts", E)))
        full_I = int(getattr(self.moe_config, "intermediate_size", intermediate_size_per_partition))
        tp_size = int(getattr(getattr(self.moe_config, "moe_parallel_config", None), "tp_size", 1))
        if tp_size != 1 and self.manifest.has_quantized_experts(self.prefix):
            raise NotImplementedError(
                f"TQ MoE {self.prefix}: compressed expert matrices are full-matrix transforms; "
                f"TP={tp_size} cannot slice their packed representation. Use TP=1 + expert parallelism."
            )
        I = int(intermediate_size_per_partition)
        if tp_size == 1:
            I = full_I
        I = ((I + 31) // 32) * 32

        self.num_experts = E
        self.global_num_experts = global_E
        self.hidden_size = H
        self.intermediate_size = I
        self.params_dtype = params_dtype
        self._role_dims: dict[str, tuple[int, int]] = {}
        self._role_batches: dict[str, int] = {}
        self._local_to_global = self._resolve_local_global_ids(layer, E, global_E)
        self._global_to_local = {g: l for l, g in enumerate(self._local_to_global)}
        self._role_is_tq_cufftdx: dict[str, tuple[bool, bool]] = {}

        # self.DL_gu = rand_signs_splitmix_fixed(H, "cuda", FIXED_SEED_D_R, out_dtype=torch.float16)
        # self.DR_gu = rand_signs_splitmix_fixed(I, "cuda", FIXED_SEED_D_L, out_dtype=torch.float32)[None, :]
        # self.DL_down = rand_signs_splitmix_fixed(I, "cuda", FIXED_SEED_D_R, out_dtype=torch.float16)
        # self.DR_down = rand_signs_splitmix_fixed(H, "cuda", FIXED_SEED_D_L, out_dtype=torch.float32)[None, :]

        device = self._tq_device

        self.DL_gu = rand_signs_splitmix_fixed(
            H, device, FIXED_SEED_D_R, out_dtype=torch.float16
        )
        self.DR_gu = rand_signs_splitmix_fixed(
            I, device, FIXED_SEED_D_L, out_dtype=torch.float32
        )[None, :]
        self.DL_down = rand_signs_splitmix_fixed(
            I, device, FIXED_SEED_D_R, out_dtype=torch.float16
        )
        self.DR_down = rand_signs_splitmix_fixed(
            H, device, FIXED_SEED_D_L, out_dtype=torch.float32
        )[None, :]
        # major, minor = torch.cuda.get_device_capability()
        # self.sm = major * 100 + minor * 10
        self._build_record_plan(H, I)
        for role, B in self._role_batches.items():
            self._register_tq_role(layer, role, B, E)
            in_f, out_f = self._role_dims[role]
            # self._role_signs[role] = (
            #     rand_signs_splitmix_fixed(
            #         in_f, "cuda", FIXED_SEED_D_R, out_dtype=torch.float16
            #     ),
            #     rand_signs_splitmix_fixed(
            #         out_f, "cuda", FIXED_SEED_D_L, out_dtype=torch.float32
            #     )[None, :],
            # )
            self._role_signs[role] = (
                rand_signs_splitmix_fixed(
                    in_f,
                    device,
                    FIXED_SEED_D_R,
                    out_dtype=torch.float16,
                ),
                rand_signs_splitmix_fixed(
                    out_f,
                    device,
                    FIXED_SEED_D_L,
                    out_dtype=torch.float32,
                )[None, :],
            )

            self._role_is_tq_cufftdx[role] = (
                can_use_tq_cufftdx(in_f, FFT_SIZES),
                can_use_tq_cufftdx(out_f, IFFT_SIZES),
            )

        self._allocate_dense_fallbacks(layer, H, I, params_dtype)

        # Consume expert HF weights ourselves. This avoids allocating native dense
        # w13/w2 tensors solely to satisfy RoutedExperts' loader.
        print(
            "[TQ MOE create_weights]",
            "prefix=", self.prefix,
            "layer=", type(layer),
            "method=", type(getattr(layer, "quant_method", None)),
            flush=True,
        )
        layer._tq_original_load_weights = layer.load_weights
        layer.load_weights = types.MethodType(_tq_routed_experts_load_weights, layer)
        print(
            "[TQ MOE loader patched]",
            "prefix=", self.prefix,
            "load_weights=", layer.load_weights,
            flush=True,
        )

        # print(f"\n=== AFTER MoE create_weights: {self.prefix} ===")

        #     print(
        #         f"{name:40s}",
        #         tuple(p.shape),
        #         p.dtype,
        #         f"{mb:.2f} MiB",
        #     )
        # print(
        #     f"[TQ MoE] {self.prefix} -> manifest={self.group_name!r} "
        #     f"localE={E}/{global_E} H={H} I={I} quant={summary} dense_fallback={dense}"
        # )

    @torch.no_grad()
    def _load_tq_role(
        self,
        layer,
        local_id: int,
        role: str,
        layer_name: str,
    ):
        tensors = self.manifest.load_tensors(
            layer_name
        )

        pfx = f"tq_{role}"

        dst = getattr(
            layer,
            f"{pfx}_packed_all",
        )[local_id]

        B = dst.shape[0]

        # New safetensors layout stores the final [B, 520] bytes directly.
        if "packed_all" in tensors:
            packed = tensors["packed_all"]
            if tuple(packed.shape) != tuple(dst.shape):
                raise RuntimeError(
                    f"TQ packed_all shape mismatch for {layer_name}: "
                    f"saved={tuple(packed.shape)} runtime={tuple(dst.shape)}"
                )
            dst.copy_(packed.to(device=dst.device, dtype=torch.uint8))
            getattr(layer, f"{pfx}_u")[local_id].copy_(
                tensors["u_W"].to(device=dst.device, dtype=torch.float32).reshape(())
            )
            getattr(layer, f"{pfx}_std")[local_id].copy_(
                tensors["std_W"].to(device=dst.device, dtype=torch.float32).reshape(())
            )
            return

        sig1 = tensors[
            "SigRec1_select_packed"
        ]
        sig2 = tensors[
            "SigRec2_select_packed"
        ]
        sig3 = tensors[
            "SigRec3_select_packed"
        ]
        sig4 = tensors[
            "SigRec4_select_packed"
        ]
        x567 = tensors["X567_packed"]

        assert sig1.shape == (B, 7)
        assert sig2.shape == (B, 12)
        assert sig3.shape == (B, 29)
        assert sig4.shape == (B, 83)
        assert x567.shape == (3 * B, 128)

        # ------------------------------------------------------------
        # Final 520-byte layout
        # ------------------------------------------------------------
        #
        # [0:8]       sig1 (7 + 1 padding)
        # [8:20]      sig2
        # [20:52]     sig3 (29 + 3 padding)
        # [52:136]    sig4 (83 + 1 padding)
        # [136:264]   X5
        # [264:392]   X6
        # [392:520]   X7
        #
        # dst was initialized with zeros, so padding bytes need no copy.

        dst[:, 0:7].copy_(
            sig1.to(
                device=dst.device,
                dtype=torch.uint8,
            )
        )

        dst[:, 8:20].copy_(
            sig2.to(
                device=dst.device,
                dtype=torch.uint8,
            )
        )

        dst[:, 20:49].copy_(
            sig3.to(
                device=dst.device,
                dtype=torch.uint8,
            )
        )

        dst[:, 52:135].copy_(
            sig4.to(
                device=dst.device,
                dtype=torch.uint8,
            )
        )

        dst[:, 136:264].copy_(
            x567[:B].to(
                device=dst.device,
                dtype=torch.uint8,
            )
        )

        dst[:, 264:392].copy_(
            x567[B:2 * B].to(
                device=dst.device,
                dtype=torch.uint8,
            )
        )

        dst[:, 392:520].copy_(
            x567[2 * B:3 * B].to(
                device=dst.device,
                dtype=torch.uint8,
            )
        )

        u_dst = getattr(
            layer,
            f"{pfx}_u",
        )[local_id]
        u_dst.copy_(
            tensors["u_W"]
            .to(
                device=u_dst.device,
                dtype=torch.float32,
            )
            .reshape(())
        )

        std_dst = getattr(
            layer,
            f"{pfx}_std",
        )[local_id]
        std_dst.copy_(
            tensors["std_W"]
            .to(
                device=std_dst.device,
                dtype=torch.float32,
            )
            .reshape(())
        )

    @torch.no_grad()
    def process_weights_after_loading(
        self,
        layer,
    ) -> None:
        start = time.monotonic()
        _tq_log(f"loading MoE experts for {self.prefix} ...")

        for lid, gid in enumerate(
            self._local_to_global
        ):
            for role, layer_name in (
                self._records
                .get(gid, {})
                .items()
            ):
                self._load_tq_role(
                    layer,
                    lid,
                    role,
                    layer_name,
                )

        _tq_log(
            f"finished loading MoE experts for {self.prefix} "
            f"in {time.monotonic() - start:.1f}s"
        )

    def _local_id_for_route(self, layer, route_id: int) -> int:
        if 0 <= route_id < self.num_experts and self.global_num_experts == self.num_experts:
            return route_id
        if route_id in self._global_to_local:
            return self._global_to_local[route_id]
        # Some vLLM prepare/finalize paths already remap ids to local physical ids.
        if 0 <= route_id < self.num_experts:
            return route_id
        return -1

    def _dense_component(
        self,
        layer,
        local_id: int,
        role: str,
        x: torch.Tensor,
    ) -> torch.Tensor:
        buffer_name = self._dense_slots[role].get(local_id)
        if buffer_name is None:
            raise RuntimeError(
                f"TQ MoE {self.prefix}: expert local={local_id} has neither "
                f"quantized nor dense {role}"
            )
        return F.linear(x, getattr(layer, buffer_name))

    def _apply_tq_role(self, layer, local_id: int, role: str, x: torch.Tensor) -> torch.Tensor:
        in_f, out_f = self._role_dims[role]
        # assert N_pad % 1024 == 0, "N_pad must be multiple of 1024"

        if x.shape[-1] != in_f:
            raise RuntimeError(
                f"TQ {self.prefix} role={role}: input width {x.shape[-1]} != saved matrix input {in_f}"
            )
        DL, DR = self._role_signs[role]
        use_cufftdx_N, use_cufftdx_M = self._role_is_tq_cufftdx[role]

        
        pfx = f"tq_{role}"
        return _tq_apply_matrix(
            x,
            packed_all=getattr(layer, f"{pfx}_packed_all")[local_id],
            u_W=getattr(layer, f"{pfx}_u")[local_id],#.to(torch.float16),
            std_W=getattr(layer, f"{pfx}_std")[local_id],#.to(torch.float16),
            DL=DL, DR=DR, 
            use_cufftdx_N = use_cufftdx_N,
            use_cufftdx_M = use_cufftdx_M,
            N=in_f, M=out_f, bias=None,
        )

    def _activation(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        act = getattr(self.moe_config, "activation", None) or getattr(self.moe_config, "hidden_act", None) or "silu"
        act_name = str(getattr(act, "value", act)).lower()
        if "gelu" in act_name:
            return F.gelu(gate) * up
        if any(x in act_name for x in ("silu", "swiglu", "silu_and_mul", "swish")):
            return F.silu(gate) * up
        raise NotImplementedError(f"TQ MoE activation {act_name!r} is not implemented")

    def _expert_forward(self, layer, local_id: int, gid: int, x: torch.Tensor) -> torch.Tensor:
        rec = self._records.get(gid, {})
        is_act_mul = bool(getattr(self.moe_config, "is_act_and_mul", True))
        if is_act_mul:
            if "gate_up" in rec:
                fused = self._apply_tq_role(layer, local_id, "gate_up", x)
                if fused.shape[-1] % 2:
                    raise RuntimeError(f"Fused gate/up output is odd: {fused.shape[-1]}")
                gate, up = fused.chunk(2, dim=-1)
            else:
                gate = self._apply_tq_role(layer, local_id, "gate", x) if "gate" in rec else self._dense_component(layer, local_id, "gate", x)
                up = self._apply_tq_role(layer, local_id, "up", x) if "up" in rec else self._dense_component(layer, local_id, "up", x)
            hidden = self._activation(gate, up)
        else:
            up = self._apply_tq_role(layer, local_id, "up", x) if "up" in rec else self._dense_component(layer, local_id, "up", x)
            act = str(getattr(self.moe_config, "activation", "silu")).lower()
            hidden = F.gelu(up) if "gelu" in act else F.silu(up)

        return self._apply_tq_role(layer, local_id, "down", hidden) if "down" in rec else self._dense_component(layer, local_id, "down", hidden)

    def _split_fused_first(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        H, I = self.hidden_size, self.intermediate_size
        w = _tq_orient_matrix(w, 2 * I, H)
        return w[:I].contiguous(), w[I:2 * I].contiguous()

    def _copy_dense_fallback(
        self,
        layer,
        gid: int,
        role: str,
        w: torch.Tensor,
    ) -> None:
        lid = self._global_to_local.get(gid, -1)
        if lid < 0:
            return

        buffer_name = self._dense_slots[role].get(lid)
        if buffer_name is None:
            # Quantized component: intentionally discard dense HF copy.
            return

        H, I = self.hidden_size, self.intermediate_size
        if role in ("gate", "up"):
            w = _tq_orient_matrix(w, I, H)
        else:
            w = _tq_orient_matrix(w, H, I)

        dst = getattr(layer, buffer_name)
        dst.copy_(
            w.to(
                device=dst.device,
                dtype=self.params_dtype,
            )
        )

    def _parse_nonfused_checkpoint_weight(self, layer, name: str) -> tuple[int, str] | None:
        qual = f"{getattr(layer, 'layer_name', self.prefix)}.{name}"
        try:
            mapping = layer.get_expert_mapping(include_fused=True)
        except Exception:
            mapping = []
        matches = [(eid, shard) for _, weight_name, eid, shard in mapping if weight_name in qual]
        if matches:
            # For a normal 2-D expert tensor this is a single logical mapping.
            eid, shard = matches[0]
            return int(eid), {"w1": "gate", "w2": "down", "w3": "up"}[shard]

        # Generic fallback for custom Transformer checkpoint layouts.
        m = re.search(r"(?:^|\.)(?:experts?\.)?(\d+)(?:\.|$)", name)
        if not m:
            m = re.search(r"expert_(\d+)", name)
        if not m:
            return None
        gid = int(m.group(1))
        for token, role in (
            ("gate_up_proj", "gate_up"), ("gate_proj", "gate"),
            ("up_proj", "up"), ("down_proj", "down"),
            (".w1", "gate"), (".w2", "down"), (".w3", "up"),
        ):
            if token in name:
                return gid, role
        return None

    @torch.no_grad()
    def _consume_hf_expert_weight(
        self,
        layer,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> bool:

        # ----------------------------------------------------------
        # Residual-checkpoint placeholder.
        #
        # make_residual_checkpoint_moe.py replaces an HF expert
        # source tensor with a zero-sized tensor only when that
        # entire source is represented by TQ records.
        #
        # There is therefore nothing to copy into dense fallback
        # storage. Just consume the checkpoint entry.
        # ----------------------------------------------------------
        if loaded_weight.numel() == 0:
            return True

        # Only consume actual expert model matrices; let the native
        # loader handle scales/biases/other tensors if a future
        # architecture adds them.
        if "weight" not in name and loaded_weight.ndim not in (2, 3):
            return False

        if loaded_weight.ndim == 3:
            # Fused checkpoint tensor [global_E, out, in].
            if loaded_weight.shape[0] < 1:
                return False

            sample = loaded_weight[0]
            role = None
            H, I = self.hidden_size, self.intermediate_size

            try:
                sample_shape = tuple(sample.shape)

                if sample_shape in ((2 * I, H), (H, 2 * I)):
                    role = "gate_up"

                elif sample_shape in ((H, I), (I, H)):
                    lname = name.lower()

                    if "down" in lname or "w2" in lname:
                        role = "down"
                    elif "gate_up" in lname or "w13" in lname:
                        role = "gate_up"
                    elif "gate" in lname or "w1" in lname:
                        role = "gate"
                    else:
                        role = "up"

            except Exception:
                role = None

            if role is None:
                return False

            for gid, w in enumerate(
                loaded_weight.unbind(0)
            ):
                if gid not in self._global_to_local:
                    continue

                if role == "gate_up":
                    gate_w, up_w = self._split_fused_first(w)

                    self._copy_dense_fallback(
                        layer,
                        gid,
                        "gate",
                        gate_w,
                    )

                    self._copy_dense_fallback(
                        layer,
                        gid,
                        "up",
                        up_w,
                    )

                else:
                    self._copy_dense_fallback(
                        layer,
                        gid,
                        role,
                        w,
                    )

            return True

        if loaded_weight.ndim != 2:
            return False

        parsed = self._parse_nonfused_checkpoint_weight(
            layer,
            name,
        )

        if parsed is None:
            return False

        gid, role = parsed

        if gid not in self._global_to_local:
            return True

        H, I = (
            self.hidden_size,
            self.intermediate_size,
        )

        # ----------------------------------------------------------
        # IMPORTANT: detect fused gate+up by SHAPE, not only by the
        # parsed projection label.
        #
        # vLLM's expert mapping may report a fused gate_up tensor via
        # the first logical shard (often w1 -> "gate", or sometimes
        # w3 -> "up").  In that case the checkpoint tensor is still:
        #
        #   [2*I, H]  or transposed [H, 2*I]
        #
        # For Qwen3.5 with I=1024, H=4096 this is exactly
        # (2048, 4096), which must be split before attempting to
        # orient/copy either individual fallback matrix.
        # ----------------------------------------------------------
        is_fused_gate_up_shape = (
            tuple(loaded_weight.shape) == (2 * I, H)
            or tuple(loaded_weight.shape) == (H, 2 * I)
        )

        if is_fused_gate_up_shape:
            gate_w, up_w = self._split_fused_first(
                loaded_weight
            )

            self._copy_dense_fallback(
                layer,
                gid,
                "gate",
                gate_w,
            )

            self._copy_dense_fallback(
                layer,
                gid,
                "up",
                up_w,
            )

            return True

        if role == "gate_up":
            # A gate_up label is only valid for a physically fused
            # matrix. Give a clearer error instead of later failing
            # inside an individual gate/up fallback copy.
            raise RuntimeError(
                "TQ checkpoint expert is labeled gate_up but its "
                f"shape is {tuple(loaded_weight.shape)}; expected "
                f"({2 * I}, {H}) or ({H}, {2 * I}). "
                f"name={name!r}, expert_id={gid}"
            )

        self._copy_dense_fallback(
            layer,
            gid,
            role,
            loaded_weight,
        )

        return True

    def _apply_routed(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
    ):
        original_shape = x.shape

        x2d = x.reshape(-1, self.hidden_size)

        out = torch.zeros(
            (x2d.shape[0], self.hidden_size),
            dtype=x2d.dtype,
            device=x2d.device,
        )

        # CUDA-graph-safe correctness-first routing.
        #
        # Avoids:
        #   torch.where()
        #   nonzero()
        #   dynamic index_select()
        #   dynamic index_add_()
        #   .item()
        #
        # All expert calls keep the fixed shape [num_tokens, hidden_size].
        for route_eid in range(self.global_num_experts):

            # Static Python-side expert mapping.
            lid = self._local_id_for_route(
                layer,
                route_eid,
            )

            if lid < 0:
                continue

            gid = self._local_to_global[lid]

            # [num_tokens, top_k]
            route_mask = (
                topk_ids == route_eid
            )

            # Combined routing weight for this expert.
            #
            # [num_tokens]
            route_weight = (
                topk_weights
                * route_mask.to(topk_weights.dtype)
            ).sum(dim=-1)

            # Mask inactive tokens BEFORE expert computation.
            #
            # This does NOT reduce matrix-multiply FLOPs, but:
            #   - preserves fixed tensor shapes
            #   - avoids meaningless nonzero activations entering the expert
            #   - can reduce work inside operations that benefit from zeros
            #
            # [num_tokens, 1]
            active = (
                route_mask
                .any(dim=-1)
                .to(x2d.dtype)
                .unsqueeze(-1)
            )

            # Fixed [num_tokens, hidden_size].
            expert_x = x2d * active

            # Fixed-shape expert execution.
            expert_out = self._expert_forward(
                layer,
                lid,
                gid,
                expert_x,
            )

            # [num_tokens, 1]
            rw = (
                route_weight
                .to(expert_out.dtype)
                .unsqueeze(-1)
            )

            # Inactive tokens contribute exactly zero.
            out.add_(
                expert_out * rw
            )

        return out.reshape(
            *original_shape[:-1],
            self.hidden_size,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._apply_routed(layer, x, topk_weights, topk_ids)

    def apply_monolithic(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        router = getattr(layer, "router", None)
        if router is None:
            raise RuntimeError(
                "TQ MoE expected modular topk routing; monolithic fallback has no router."
            )
        try:
            topk_weights, topk_ids = router.select_experts(x, router_logits, input_ids=input_ids)
        except TypeError:
            topk_weights, topk_ids = router.select_experts(x, router_logits)
        return self._apply_routed(layer, x, topk_weights, topk_ids)



# ---------------------------------------------------------------------------
# TorchDynamo bridge for closed/Cython wheels
def _install_tq_dynamo_dense_shims() -> None:
    """Install real-Python apply methods so Dynamo can trace closed Cython wheels."""
    ns = {"torch": torch}
    exec(_TQ_DYNAMO_SHIM_SOURCE, ns, ns)
    globals()["_tq_apply_matrix_traceable"] = ns["_tq_apply_matrix_traceable"]
    globals()["_tq_mimo_deinterleave_traceable"] = ns["_tq_mimo_deinterleave_traceable"]
    TqLinearMethod.apply = ns["_tq_linear_apply"]
    TqQKVMethod.apply = ns["_tq_qkv_apply"]
    TqGateUpMethod.apply = ns["_tq_gate_up_apply"]

_TQ_DYNAMO_SHIM_SOURCE = 'def _tq_apply_matrix_traceable(input, packed_all, u_W, std_W, DL, DR,\n                               use_cufftdx_N, use_cufftdx_M, N, M, bias=None):\n    in_dtype = input.dtype\n    in_shape = input.shape\n    x_in = input.reshape(-1, input.shape[-1]).to(torch.float16)\n    x, input_sum = torch.ops.my_qlinear.forward_fft(x_in, N, DL, use_cufftdx_N)\n    x = x.contiguous()\n    x = torch.ops.my_qlinear.forward_pass(packed_all, x, N, M)\n    x = torch.ops.my_qlinear.reverse_ifft(x, DR, std_W, input_sum, u_W, bias, use_cufftdx_M)\n    x = x.reshape(*in_shape[:-1], M)\n    return x.to(in_dtype)\n\ndef _tq_mimo_deinterleave_traceable(y, spec):\n    n_kv = int(spec["num_kv_heads"])\n    qpg = int(spec["q_rows_per_group"])\n    hd = int(spec["head_dim"])\n    vhd = int(spec["v_head_dim"])\n    rpg = int(spec["rows_per_group"])\n    leading = y.shape[:-1]\n    g = y.reshape(*leading, n_kv, rpg)\n    q = g[..., :, :qpg].reshape(*leading, n_kv * qpg)\n    k = g[..., :, qpg:qpg + hd].reshape(*leading, n_kv * hd)\n    v = g[..., :, qpg + hd:qpg + hd + vhd].reshape(*leading, n_kv * vhd)\n    return torch.cat((q, k, v), dim=-1)\n\ndef _tq_linear_apply(self, layer, input, bias=None):\n    out = _tq_apply_matrix_traceable(input, layer.packed_all, layer.u_W, layer.std_W,\n        self.DL, self.DR, self.use_cufftdx_N, self.use_cufftdx_M, self.N, self.M, bias)\n    spec = self._mimo_qkv_spec\n    if spec is not None:\n        out = _tq_mimo_deinterleave_traceable(out, spec)\n    return out\n\ndef _tq_qkv_apply(self, layer, input, bias=None):\n    q = _tq_apply_matrix_traceable(input, layer.tq_q_packed_all, layer.tq_q_u, layer.tq_q_std,\n        self._q_DL, self._q_DR, self._q_use_cufftdx_N, self._q_use_cufftdx_M, self.N, self.q_M, None)\n    k = _tq_apply_matrix_traceable(input, layer.tq_k_packed_all, layer.tq_k_u, layer.tq_k_std,\n        self._k_DL, self._k_DR, self._k_use_cufftdx_N, self._k_use_cufftdx_M, self.N, self.k_M, None)\n    v = _tq_apply_matrix_traceable(input, layer.tq_v_packed_all, layer.tq_v_u, layer.tq_v_std,\n        self._v_DL, self._v_DR, self._v_use_cufftdx_N, self._v_use_cufftdx_M, self.N, self.v_M, None)\n    out = torch.cat((q, k, v), dim=-1)\n    if bias is not None:\n        out = out + bias\n    return out\n\ndef _tq_gate_up_apply(self, layer, input, bias=None):\n    gate = _tq_apply_matrix_traceable(input, layer.tq_gate_packed_all, layer.tq_gate_u, layer.tq_gate_std,\n        self._gate_DL, self._gate_DR, self._gate_use_cufftdx_N, self._gate_use_cufftdx_M, self.N, self.gate_M, None)\n    up = _tq_apply_matrix_traceable(input, layer.tq_up_packed_all, layer.tq_up_u, layer.tq_up_std,\n        self._up_DL, self._up_DR, self._up_use_cufftdx_N, self._up_use_cufftdx_M, self.N, self.up_M, None)\n    out = torch.cat((gate, up), dim=-1)\n    if bias is not None:\n        out = out + bias\n    return out\n'

_install_tq_dynamo_dense_shims()
