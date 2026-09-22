#!/usr/bin/env python3

import argparse
import json
import os
import re
import shutil
import tarfile
import zipfile
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import snapshot_download, login, HfApi


WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")

QUANT_DIR_NAME = "quantization_data"
TQ_SHARD_INDEX_NAME = "tq_shards.json"

# Process-local cache for tq_shards.json metadata.
# build_replacement_plan() touches tens of thousands of records; without
# this cache the old code reparsed the whole JSON index once per record.
_TQ_META_INDEX_CACHE = {}
TQ_SHARD_TARGET_MB = int(os.environ.get("TQ_SHARD_TARGET_MB", "5120"))
TQ_SHARD_TARGET_BYTES = TQ_SHARD_TARGET_MB * 1024 * 1024


def sanitize_layer_name(layer_name: str) -> str:
    return layer_name.replace(".", "_").replace("/", "_")


def load_tq_manifest(tq_dir: Path) -> list[str]:
    layers_file = tq_dir / "quantization_data" / "_layers.json"
    if not layers_file.exists():
        raise FileNotFoundError(layers_file)
    with open(layers_file, "r") as f:
        return json.load(f)


def detect_multimodal_target(tq_dir: Path) -> bool:
    """Detect multimodal wrapper from the actual top-level HF config.

    Supports Qwen3.5 and Qwen4Exp-style ConditionalGeneration models.
    The strongest generic signals are:
      * a non-empty top-level vision_config;
      * architecture name ending in ForConditionalGeneration;
      * language_model_only explicitly false.
    """
    config_file = tq_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(config_file)

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    architectures = cfg.get("architectures") or []
    if isinstance(architectures, str):
        architectures = [architectures]

    vision_config = cfg.get("vision_config")
    language_model_only = cfg.get("language_model_only")

    if isinstance(vision_config, dict) and vision_config:
        return True

    if any(str(a).endswith("ForConditionalGeneration") for a in architectures):
        return True

    if language_model_only is False:
        return True

    return False


def get_tq_model_family(tq_dir: Path) -> str:
    config_file = tq_dir / "config.json"
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return str(cfg.get("model_type") or "")


def tq_layer_name_to_runtime(
    layer_name: str,
    *,
    multimodal: bool,
) -> str:
    """Map TQ logical names between standalone text and multimodal wrappers.

    Qwen3.5 and Qwen4Exp ConditionalGeneration checkpoints both use an outer
    language-model wrapper. Quantizer records may come from either the inner HF
    text namespace or an already-normalized runtime namespace.
    """
    name = str(layer_name)

    if not multimodal:
        mm_body = "model.language_model.model."
        if name.startswith(mm_body):
            return "model." + name[len(mm_body):]

        short_mm = "model.language_model."
        if name.startswith(short_mm):
            rest = name[len(short_mm):]
            if rest.startswith("lm_head"):
                return rest
            if rest.startswith(("layers.", "embed_tokens.", "norm.", "mtp.")):
                return "model." + rest
        return name

    # Vision stays under the outer multimodal model.
    if name.startswith("model.visual.") or name.startswith("visual."):
        return name if name.startswith("model.") else "model." + name

    # Already-normalized multimodal body.
    if name.startswith("model.language_model.model."):
        return name

    if name.startswith("model.language_model.lm_head"):
        return name

    # Quantizer commonly sees the inner text model as:
    #   model.language_model.layers.X...
    # Native multimodal runtime expects:
    #   model.language_model.model.layers.X...
    short_mm = "model.language_model."
    if name.startswith(short_mm):
        rest = name[len(short_mm):]
        if rest.startswith(("layers.", "embed_tokens.", "norm.", "mtp.")):
            return "model.language_model.model." + rest
        return name

    # Standalone text-model logical names.
    if name.startswith("model."):
        return "model.language_model.model." + name[len("model."):]

    if name == "lm_head" or name.startswith("lm_head."):
        return "model.language_model." + name

    return name


def rewrite_tq_manifest_for_runtime(
    tq_dir: Path,
    *,
    multimodal: bool,
) -> dict:
    """Canonicalize TQ logical names and safely deduplicate aliases.

    Re-quantization of an already-converted directory can leave _layers.json
    containing both forms of the same logical target, for example:

      model.language_model.layers.0....
      model.language_model.model.layers.0....

    Both canonicalize to the same native-vLLM runtime name. This function now
    treats those as aliases rather than a fatal collision.

    The safetensor index is canonicalized at the same time. If multiple alias
    keys resolve to the same canonical name:
      * identical index records are trivially deduplicated;
      * if only one alias has an index record, that record is preserved;
      * if two different records claim the same canonical target, we fail
        loudly rather than silently choosing one.

    Safetensor tensor storage keys themselves are never rewritten.
    """
    quant_dir = tq_dir / QUANT_DIR_NAME
    layers_file = quant_dir / "_layers.json"

    with open(layers_file, "r", encoding="utf-8") as f:
        old_layers = json.load(f)

    if not isinstance(old_layers, list):
        raise TypeError(f"{layers_file} must contain a JSON list")

    old_layers = [str(x) for x in old_layers]

    mapping = {
        old: tq_layer_name_to_runtime(old, multimodal=multimodal)
        for old in old_layers
    }

    # Build a stable, deduplicated canonical manifest.
    new_layers = []
    canonical_sources = defaultdict(list)
    seen_canonical = set()

    for old in old_layers:
        new = mapping[old]
        canonical_sources[new].append(old)
        if new not in seen_canonical:
            seen_canonical.add(new)
            new_layers.append(new)

    duplicate_aliases = {
        new: olds
        for new, olds in canonical_sources.items()
        if len(olds) > 1
    }

    shard_index_file = quant_dir / TQ_SHARD_INDEX_NAME
    rewritten_locations = None
    deduped_index_aliases = 0

    if shard_index_file.exists():
        with open(shard_index_file, "r", encoding="utf-8") as f:
            shard_index = json.load(f)

        locations = shard_index.get("layers")
        if not isinstance(locations, dict):
            raise RuntimeError(
                f"{shard_index_file} does not contain a dict at key 'layers'"
            )

        # Canonicalize every existing index key, including entries that may not
        # currently appear in _layers.json. This keeps the index self-consistent.
        rewritten_locations = {}

        for logical_name, location in locations.items():
            canonical = tq_layer_name_to_runtime(
                str(logical_name),
                multimodal=multimodal,
            )

            existing = rewritten_locations.get(canonical)
            if existing is None:
                rewritten_locations[canonical] = location
                continue

            # Same canonical target appears under multiple aliases.
            if existing == location:
                deduped_index_aliases += 1
                continue

            # It is possible for JSON object ordering to differ while the record
            # is semantically identical; compare normalized JSON too.
            if json.dumps(existing, sort_keys=True) == json.dumps(
                location,
                sort_keys=True,
            ):
                deduped_index_aliases += 1
                continue

            # Never silently discard two genuinely different quantized records
            # for one runtime parameter.
            raise RuntimeError(
                "Conflicting TQ safetensor records canonicalize to the same "
                "runtime layer:\n"
                f"  canonical={canonical}\n"
                f"  first={json.dumps(existing, sort_keys=True)[:1000]}\n"
                f"  second={json.dumps(location, sort_keys=True)[:1000]}"
            )

        # Every canonical manifest record must exist in the safetensor index.
        missing_from_index = [
            name for name in new_layers
            if name not in rewritten_locations
        ]
        if missing_from_index:
            raise RuntimeError(
                f"{len(missing_from_index)} canonical _layers.json records "
                "were not found in tq_shards.json after alias normalization:\n  "
                + "\n  ".join(missing_from_index[:100])
            )

        shard_index["layers"] = rewritten_locations

        # Recompute shard count from the actual merged index.
        shard_names = {
            Path(str(v["filename"])).name
            for v in rewritten_locations.values()
            if isinstance(v, dict) and v.get("filename")
        }
        shard_index["num_shards"] = len(shard_names)
        shard_index["format"] = "tq_safetensors_shards_v1"

        tq_shard_index_file = quant_dir / TQ_SHARD_INDEX_NAME
        tmp_index = tq_shard_index_file.with_suffix(".json.tmp")
        with open(tmp_index, "w", encoding="utf-8") as f:
            json.dump(shard_index, f, indent=2, sort_keys=True)
        tmp_index.replace(tq_shard_index_file)


    else:
        # Legacy .pt-only layout. Rename physical records to canonical names,
        # but dedupe aliases if the canonical destination already exists.
        missing_record_files = []

        for canonical, aliases in canonical_sources.items():
            target_file = quant_dir / f"{sanitize_layer_name(canonical)}.pt"

            if target_file.exists():
                # Canonical record already exists. Remove only exact alias files
                # after checking they are separate paths; the canonical record wins.
                for alias in aliases:
                    alias_file = quant_dir / f"{sanitize_layer_name(alias)}.pt"
                    if alias_file != target_file and alias_file.exists():
                        alias_file.unlink()
                continue

            source_file = None
            for alias in aliases:
                candidate = quant_dir / f"{sanitize_layer_name(alias)}.pt"
                if candidate.exists():
                    source_file = candidate
                    break

                # Historical name aliases.
                for alt in _tq_index_name_candidates(
                    alias,
                    multimodal=multimodal,
                ):
                    candidate = quant_dir / f"{sanitize_layer_name(alt)}.pt"
                    if candidate.exists():
                        source_file = candidate
                        break
                if source_file is not None:
                    break

            if source_file is None:
                missing_record_files.append(
                    {
                        "canonical": canonical,
                        "aliases": aliases,
                    }
                )
                continue

            if source_file != target_file:
                source_file.rename(target_file)

        if missing_record_files:
            raise RuntimeError(
                "TQ uses the legacy per-layer .pt layout, but some canonical "
                "records could not be found:\n"
                + json.dumps(missing_record_files[:100], indent=2)
            )

    # Rewrite _layers.json LAST, after the index/physical records are valid.
    tmp_layers = layers_file.with_suffix(".json.tmp")
    with open(tmp_layers, "w", encoding="utf-8") as f:
        json.dump(new_layers, f, indent=2)
    tmp_layers.replace(layers_file)

    changed = [
        (old, mapping[old])
        for old in old_layers
        if old != mapping[old]
    ]

    print(
        f"[manifest] canonical records: {len(new_layers)} "
        f"(from {len(old_layers)} manifest entries)"
    )
    if duplicate_aliases:
        duplicate_extra = sum(len(v) - 1 for v in duplicate_aliases.values())
        print(
            f"[manifest] deduplicated {duplicate_extra} duplicate alias "
            f"entries across {len(duplicate_aliases)} runtime targets"
        )
    if deduped_index_aliases:
        print(
            f"[manifest] deduplicated {deduped_index_aliases} duplicate "
            "safetensor index aliases"
        )

    return {
        "changed": changed,
        "count": len(new_layers),
        "multimodal": multimodal,
        "duplicate_alias_targets": len(duplicate_aliases),
        "duplicate_alias_entries_removed": sum(
            len(v) - 1 for v in duplicate_aliases.values()
        ),
        "deduped_index_aliases": deduped_index_aliases,
    }



def load_quant_record_meta(tq_dir: Path, layer_name: str) -> dict:
    """
    Load metadata for one logical TQ record.

    Preferred/native layout:
      quantization_data/tq_shards.json contains each layer's metadata inline.

    Backward-compatible layout:
      quantization_data/<sanitized-layer>.pt stores {"meta": ...}.
    """
    quant_dir = tq_dir / "quantization_data"

    # Preferred sharded layout. tq_quantizer.py also uses the metadata embedded in this
    # JSON index, so reading it here avoids deserializing a large TQ shard.
    shard_index_file = quant_dir / "tq_shards.json"

    if shard_index_file.exists():
        # Do not json.load() tq_shards.json once per logical layer.
        # Qwen3.8 has ~50k TQ records, so the old implementation reparsed the
        # complete index ~50k times and appeared hung after conversion.
        cache_key = str(shard_index_file.resolve())
        stat = shard_index_file.stat()
        stamp = (int(stat.st_mtime_ns), int(stat.st_size))

        cached = _TQ_META_INDEX_CACHE.get(cache_key)
        if cached is None or cached[0] != stamp:
            print(
                f"[metadata] loading TQ shard index once: {shard_index_file}",
                flush=True,
            )
            with open(shard_index_file, "r", encoding="utf-8") as f:
                shard_index = json.load(f)

            layers_map = shard_index.get("layers", {})
            if not isinstance(layers_map, dict):
                raise RuntimeError(
                    f"{shard_index_file} does not contain a dict at key 'layers'"
                )

            _TQ_META_INDEX_CACHE[cache_key] = (stamp, layers_map)
        else:
            layers_map = cached[1]

        location = layers_map.get(layer_name)
        if isinstance(location, dict):
            meta = location.get("meta")
            if isinstance(meta, dict):
                return dict(meta)

    # Legacy/incremental quantizer output.
    pt_file = quant_dir / f"{sanitize_layer_name(layer_name)}.pt"
    if pt_file.exists():
        record = torch.load(
            pt_file,
            map_location="cpu",
            weights_only=False,
        )
        meta = dict(record.get("meta", {}))
        del record
        return meta

    raise FileNotFoundError(
        "Quantized layer is listed in _layers.json but its metadata "
        "could not be found in either tq_shards.json or a per-layer .pt file:\n"
        f"  layer={layer_name}\n"
        f"  shard_index={shard_index_file}\n"
        f"  legacy_file={pt_file}"
    )


def looks_like_expert_record(layer_name: str) -> bool:
    lower = layer_name.lower()
    return (
        ".experts." in lower
        or ".expert_" in lower
        or ".moe." in lower
        or ".routed_experts." in lower
    )


_NATIVE_PACKED_SUFFIXES = (
    ".qkv_proj",
    ".gate_up_proj",
    ".in_proj_qkvz",
    ".in_proj_ba",
)


def looks_like_native_packed_record(layer_name: str) -> bool:
    """
    Native-vLLM packed records usually do not have a same-named tensor in the
    original HF checkpoint. They must carry source_state_keys provenance.
    """
    lower = layer_name.lower()
    return any(lower.endswith(suffix) for suffix in _NATIVE_PACKED_SUFFIXES)


def build_replacement_plan(tq_dir: Path, layer_names: list[str]) -> dict:
    """
    Determine which original HF tensors are fully represented by TQ.

    Native-vLLM TQ records may represent MULTIPLE original HF tensors.

    Examples:
      qkv_proj
        <- q_proj.weight
        <- k_proj.weight
        <- v_proj.weight

      gate_up_proj
        <- gate_proj.weight
        <- up_proj.weight

      in_proj_qkvz
        <- in_proj_qkv.weight
        <- in_proj_z.weight

    The native quantizer stores these in:
      source_state_keys: [...]
      source_shapes: [...]

    For ordinary rank-2 source tensors, slice_index is None and every listed
    source tensor is fully replaced.

    For rank-3 MoE source tensors [E, M, N], one TQ record may represent one
    expert slice from one OR multiple source tensors. A rank-3 source is removed
    only when every expert slice 0..E-1 is represented by TQ.

    Partially covered rank-3 sources are emitted as exact missing-expert 2-D fallback tensors.
    """

    full_replaced: set[str] = set()
    sliced_coverage: dict[str, set[int]] = defaultdict(set)
    source_shapes: dict[str, tuple[int, ...]] = {}

    records_without_provenance = []
    native_records_without_provenance = []
    multi_source_records = 0
    embedding_shard_records = 0
    embedding_shard_sources: set[str] = set()

    total_plan_records = len(layer_names)
    print(
        f"[replacement-plan] analyzing {total_plan_records} TQ records...",
        flush=True,
    )

    for plan_i, layer_name in enumerate(layer_names, 1):
        if plan_i == 1 or plan_i % 10000 == 0 or plan_i == total_plan_records:
            print(
                f"[replacement-plan] {plan_i}/{total_plan_records}",
                flush=True,
            )

        meta = load_quant_record_meta(tq_dir, layer_name)

        slice_index = meta.get("slice_index")

        # --------------------------------------------------------------
        # New/native format: one logical TQ matrix can come from several
        # HF checkpoint tensors.
        # --------------------------------------------------------------
        source_keys_raw = meta.get("source_state_keys")
        source_shapes_raw = meta.get("source_shapes")

        source_keys: list[str] = []
        aligned_shapes: list[tuple[int, ...] | None] = []

        if isinstance(source_keys_raw, (list, tuple)) and source_keys_raw:
            source_keys = [str(x) for x in source_keys_raw if x]

            if len(source_keys) > 1:
                multi_source_records += 1

            if isinstance(source_shapes_raw, (list, tuple)):
                for shape in source_shapes_raw:
                    if shape is None:
                        aligned_shapes.append(None)
                    else:
                        aligned_shapes.append(tuple(int(x) for x in shape))

            # Be strict about alignment if source_shapes was supplied.
            if aligned_shapes and len(aligned_shapes) != len(source_keys):
                raise RuntimeError(
                    f"TQ metadata source_state_keys/source_shapes length mismatch "
                    f"for {layer_name!r}: keys={len(source_keys)} "
                    f"shapes={len(aligned_shapes)}"
                )

        # --------------------------------------------------------------
        # Backward-compatible single-source metadata.
        # --------------------------------------------------------------
        if not source_keys:
            source_key = meta.get("source_state_key")
            source_shape = meta.get("source_shape")

            if source_key:
                source_keys = [str(source_key)]
                aligned_shapes = [
                    None
                    if source_shape is None
                    else tuple(int(x) for x in source_shape)
                ]

        target_kind = str(meta.get("target_kind") or "")
        if target_kind in {"tq_embedding_shard", "tq_ngram_chunked_shard"}:
            if len(source_keys) != 1:
                raise RuntimeError(
                    f"TQ embedding shard {layer_name!r} must have exactly one "
                    f"source_state_key, got {source_keys}"
                )
            if not aligned_shapes:
                aligned_shapes = [None]
            source_key = source_keys[0]
            shape = aligned_shapes[0]
            if shape is None or len(shape) != 2:
                raise RuntimeError(
                    f"TQ embedding shard {layer_name!r} has invalid source_shape={shape}"
                )
            if (
                ".ngram_embedding.shard_" not in source_key
                or not source_key.endswith(".weight")
            ):
                raise RuntimeError(
                    f"TQ embedding shard {layer_name!r} points to unexpected "
                    f"HF source key {source_key!r}"
                )
            matrix_shape = meta.get("matrix_shape")
            if not isinstance(matrix_shape, (list, tuple)) or len(matrix_shape) != 2:
                raise RuntimeError(
                    f"TQ embedding shard {layer_name!r} lacks matrix_shape metadata"
                )
            src_rows, src_dim = map(int, shape)
            tq_m, tq_n = map(int, matrix_shape)

            if target_kind == "tq_ngram_chunked_shard":
                chunk_rows = int(meta.get("ngram_chunk_rows") or tq_n)
                embed_dim = int(meta.get("ngram_embedding_dim") or tq_m)
                num_chunks = int(meta.get("ngram_num_chunks") or 0)
                valid_rows = int(meta.get("ngram_valid_rows") or src_rows)
                expected_chunks = (src_rows + chunk_rows - 1) // chunk_rows
                if (tq_m, tq_n) != (src_dim, chunk_rows):
                    raise RuntimeError(
                        f"TQ n-gram shard {layer_name!r} has bad matrix geometry: "
                        f"source={shape}, matrix_shape={matrix_shape}, "
                        f"chunk_rows={chunk_rows}"
                    )
                if embed_dim != src_dim or valid_rows != src_rows:
                    raise RuntimeError(
                        f"TQ n-gram shard {layer_name!r} has inconsistent metadata: "
                        f"embedding_dim={embed_dim}, valid_rows={valid_rows}, "
                        f"source={shape}"
                    )
                if num_chunks and num_chunks != expected_chunks:
                    raise RuntimeError(
                        f"TQ n-gram shard {layer_name!r} has ngram_num_chunks="
                        f"{num_chunks}, expected {expected_chunks}"
                    )
            else:
                if (tq_m, tq_n) != (src_dim, src_rows):
                    raise RuntimeError(
                        f"TQ embedding shard {layer_name!r} was not stored as source.T: "
                        f"source={shape}, matrix_shape={matrix_shape}"
                    )

            g0 = meta.get("embedding_row_start")
            g1 = meta.get("embedding_row_end")
            if g0 is not None or g1 is not None:
                if g0 is None or g1 is None or int(g1) - int(g0) != src_rows:
                    raise RuntimeError(
                        f"TQ embedding shard {layer_name!r} has invalid global row "
                        f"metadata [{g0}:{g1}] for {src_rows} source rows"
                    )
            if source_key in embedding_shard_sources:
                raise RuntimeError(
                    f"Duplicate TQ embedding ownership for HF source {source_key!r}"
                )
            embedding_shard_records += 1
            embedding_shard_sources.add(source_key)

        if source_keys:
            if not aligned_shapes:
                aligned_shapes = [None] * len(source_keys)

            # A native fused record owns every source listed in provenance.
            for source_key, shape in zip(source_keys, aligned_shapes):
                if shape is not None:
                    old_shape = source_shapes.get(source_key)
                    if old_shape is not None and old_shape != shape:
                        raise RuntimeError(
                            f"Inconsistent source shape metadata for {source_key!r}: "
                            f"{old_shape} vs {shape} (record={layer_name!r})"
                        )
                    source_shapes[source_key] = shape

                if slice_index is None:
                    full_replaced.add(source_key)
                else:
                    sliced_coverage[source_key].add(int(slice_index))

            continue

        # --------------------------------------------------------------
        # Legacy fallback.
        #
        # Safe only for an ordinary non-MoE, non-packed layer whose TQ
        # record name still equals its HF checkpoint module name.
        #
        # Native packed names MUST carry provenance because e.g.
        # qkv_proj.weight generally does not exist in the source HF checkpoint.
        # --------------------------------------------------------------
        if looks_like_native_packed_record(layer_name):
            native_records_without_provenance.append(layer_name)
            continue

        if not looks_like_expert_record(layer_name):
            full_replaced.add(layer_name + ".weight")
        else:
            records_without_provenance.append(layer_name)

    fully_covered_fused = set()
    partially_covered_fused = {}

    for source_key, slices in sliced_coverage.items():
        # If some record says the WHOLE tensor is replaced, that takes precedence.
        if source_key in full_replaced:
            continue

        shape = source_shapes.get(source_key)

        if not shape:
            partially_covered_fused[source_key] = {
                "reason": "missing source_shape/source_shapes metadata",
                "slices": sorted(slices),
            }
            continue

        if len(shape) != 3:
            partially_covered_fused[source_key] = {
                "reason": (
                    f"slice_index present but source shape={shape} is not rank-3"
                ),
                "slices": sorted(slices),
            }
            continue

        num_experts = int(shape[0])
        expected = set(range(num_experts))

        if slices == expected:
            full_replaced.add(source_key)
            fully_covered_fused.add(source_key)
        else:
            partially_covered_fused[source_key] = {
                "shape": shape,
                "quantized": len(slices),
                "total": num_experts,
                "missing": sorted(expected - slices),
            }

    if embedding_shard_records:
        if embedding_shard_records != len(embedding_shard_sources):
            raise RuntimeError(
                "TQ n-gram physical-shard accounting mismatch: "
                f"records={embedding_shard_records}, "
                f"distinct_sources={len(embedding_shard_sources)}"
            )
        qwen38_sources = [
            k for k in embedding_shard_sources
            if (
                "layers.1.ple.ple_embedding.ngram_embedding.shard_" in k
                and k.endswith(".weight")
            )
        ]
        if qwen38_sources and len(qwen38_sources) != 128:
            raise RuntimeError(
                "Detected Qwen3.8-Flash-Next n-gram TQ records but found "
                f"{len(qwen38_sources)}/128 physical source shards. "
                "Finish quantizing all n-gram shards before building."
            )

    if native_records_without_provenance:
        raise RuntimeError(
            "Native-vLLM TQ records were found without source_state_keys "
            "provenance. The residual builder cannot safely determine which HF "
            "checkpoint tensors to remove.\n"
            "Re-quantize these layers with the native-vLLM tq_quantizer.py.\n"
            "Examples:\n  "
            + "\n  ".join(sorted(native_records_without_provenance)[:50])
        )

    return {
        "full_replaced": full_replaced,
        "fully_covered_fused": fully_covered_fused,
        "partially_covered_fused": partially_covered_fused,
        "records_without_provenance": records_without_provenance,
        "native_records_without_provenance": native_records_without_provenance,
        "multi_source_records": multi_source_records,
        "embedding_shard_records": embedding_shard_records,
        "embedding_shard_sources": embedding_shard_sources,
    }


def find_safetensors(base_dir: Path) -> list[Path]:
    index_file = base_dir / "model.safetensors.index.json"

    if index_file.exists():
        with open(index_file, "r") as f:
            index = json.load(f)
        return sorted(
            {base_dir / filename for filename in index["weight_map"].values()}
        )

    single = base_dir / "model.safetensors"
    if single.exists():
        return [single]

    files = sorted(base_dir.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"No safetensors checkpoint found in {base_dir}")
    return files


def copy_non_weight_files(src: Path, dst: Path) -> None:
    """
    Add tokenizer/generation metadata to the existing TQ directory.

    Existing files are preserved, especially config.json.
    quantization_data/ is never copied or touched.
    """
    for item in src.iterdir():
        if item.is_dir():
            continue

        name = item.name
        if name == "model.safetensors.index.json":
            continue
        if name.endswith(WEIGHT_SUFFIXES):
            continue

        target = dst / name
        if target.exists():
            continue

        shutil.copy2(item, target)


def remove_existing_residual_weights(tq_dir: Path) -> None:
    for p in tq_dir.glob("model*.safetensors"):
        if p.is_file():
            p.unlink()

    index_file = tq_dir / "model.safetensors.index.json"
    if index_file.exists():
        index_file.unlink()


def resolve_base_model(base_model: str) -> Path:
    candidate = Path(base_model)

    if candidate.exists():
        base_dir = candidate.resolve()
        print("Using local base checkpoint:", base_dir)
        return base_dir

    print("Downloading/resolving base checkpoint:", base_model)
    return Path(snapshot_download(repo_id=base_model))

def hf_key_to_native_vllm_key(
    key: str,
    *,
    multimodal: bool = False,
) -> str | None:
    """Normalize source HF checkpoint keys for the selected runtime wrapper.

    For multimodal ConditionalGeneration models (including Qwen3.5 and
    Qwen4Exp), preserve the original multimodal language/vision namespaces.
    For text-only serving, remove the language_model wrapper and drop vision.
    """
    if multimodal:
        if key.startswith("model.visual."):
            return key
        if key.startswith("visual."):
            return "model." + key
        if key.startswith("model.language_model."):
            return key
        if key.startswith("language_model."):
            return "model." + key
        return key

    if key.startswith("model.visual.") or key.startswith("visual."):
        return None

    prefix = "model.language_model."
    if key.startswith(prefix):
        return "model." + key[len(prefix):]

    prefix = "language_model."
    if key.startswith(prefix):
        return key[len(prefix):]

    return key


def _tq_sparse_expert_key(output_key: str, expert_id: int) -> str:
    """Create a routed-expert key that PcqMoEMethod can parse as a 2-D fallback.

    Examples:
      ...experts.down_proj
        -> ...experts.down_proj.expert_0089.weight

      ...experts.gate_up_proj
        -> ...experts.gate_up_proj.expert_0089.weight

    PcqMoEMethod's generic 2-D parser recognizes both `expert_####` and the
    projection role token, so these weights fill only the exact missing dense
    fallback buffers.
    """
    return f"{output_key}.expert_{int(expert_id):04d}.weight"


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)


def write_residual_checkpoint_streaming(
    *,
    base_dir: Path,
    tq_dir: Path,
    replaced_keys: set[str],
    partially_covered_fused: dict[str, dict],
    multimodal: bool,
) -> dict:
    """Write residual shards incrementally, with sparse partial-MoE fallback.

    Two disk-saving changes are implemented:

    1. Partially covered rank-3 MoE source tensors are no longer retained in
       full. The original fused source key is emitted as a zero-size placeholder
       and only the exact missing expert slices are emitted as independent 2-D
       keys (`...expert_####.weight`). PcqMoEMethod already knows how to consume
       these sparse expert keys.

    2. Each residual shard is written directly into the final TQ directory via
       a one-shard `.part` file and atomic rename. We no longer accumulate all
       94 reconstructed shards under `.tq_residual_tmp`.

    The checkpoint is marked incomplete until the final index is written.
    """
    shards = find_safetensors(base_dir)
    print("Base shards:", len(shards))

    total_shards = len(shards)

    # Remove stale residual output from a previous successful/partial build.
    # quantization_data/ is untouched.
    remove_existing_residual_weights(tq_dir)

    marker = tq_dir / ".tq_residual_in_progress"
    marker.write_text(
        "Residual reconstruction is in progress. Do not serve this directory.\n",
        encoding="utf-8",
    )

    original_bytes = 0
    residual_bytes = 0
    removed_bytes = 0
    dropped_bytes = 0
    sparse_saved_bytes = 0
    sparse_original_bytes = 0

    placeholder_count = 0
    sparse_expert_count = 0
    seen_replaced = set()
    seen_partial = set()
    weight_map: dict[str, str] = {}

    try:
        for shard_i, shard in enumerate(shards, start=1):
            print(f"[{shard_i}/{total_shards}] Scanning {shard.name}")

            shard_state: dict[str, torch.Tensor] = {}

            with safe_open(shard, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    nbytes = tensor.numel() * tensor.element_size()
                    original_bytes += nbytes

                    output_key = hf_key_to_native_vllm_key(
                        key,
                        multimodal=multimodal,
                    )

                    # Text-only mode may discard vision.
                    if output_key is None:
                        dropped_bytes += nbytes
                        continue

                    if output_key in shard_state:
                        raise RuntimeError(
                            "Checkpoint key normalization produced a duplicate key:\n"
                            f"  source={key}\n"
                            f"  output={output_key}"
                        )

                    # ------------------------------------------------------
                    # Fully TQ-owned source tensor.
                    # ------------------------------------------------------
                    if key in replaced_keys:
                        shard_state[output_key] = torch.empty(
                            0,
                            dtype=torch.uint8,
                        )
                        removed_bytes += nbytes
                        placeholder_count += 1
                        seen_replaced.add(key)
                        continue

                    # ------------------------------------------------------
                    # Partially covered fused MoE tensor.
                    #
                    # Old behavior:
                    #   keep the ENTIRE [E, O, I] tensor dense.
                    #
                    # New behavior:
                    #   placeholder original key
                    #   + only missing expert slices as 2-D fallback keys.
                    # ------------------------------------------------------
                    partial = partially_covered_fused.get(key)
                    if partial is not None and tensor.ndim == 3:
                        missing = [
                            int(x)
                            for x in partial.get("missing", [])
                        ]

                        expected_shape = tuple(
                            int(x) for x in partial.get("shape", tensor.shape)
                        )
                        if tuple(tensor.shape) != expected_shape:
                            raise RuntimeError(
                                "Partial MoE source shape mismatch:\n"
                                f"  key={key}\n"
                                f"  checkpoint={tuple(tensor.shape)}\n"
                                f"  metadata={expected_shape}"
                            )

                        # Consume the original fused source without allocating it
                        # in dense fallback storage.
                        shard_state[output_key] = torch.empty(
                            0,
                            dtype=torch.uint8,
                        )
                        placeholder_count += 1
                        seen_partial.add(key)
                        sparse_original_bytes += nbytes

                        for expert_id in missing:
                            if not (0 <= expert_id < int(tensor.shape[0])):
                                raise RuntimeError(
                                    f"Bad missing expert id {expert_id} for "
                                    f"{key} with {tensor.shape[0]} experts"
                                )

                            sparse_key = _tq_sparse_expert_key(
                                output_key,
                                expert_id,
                            )
                            if sparse_key in shard_state:
                                raise RuntimeError(
                                    f"Duplicate sparse fallback key: {sparse_key}"
                                )

                            expert_tensor = tensor[expert_id].contiguous()
                            shard_state[sparse_key] = expert_tensor

                            eb = (
                                expert_tensor.numel()
                                * expert_tensor.element_size()
                            )
                            residual_bytes += eb
                            sparse_saved_bytes += eb
                            sparse_expert_count += 1

                        # The quantized expert slices are removed from residual.
                        removed_bytes += max(
                            0,
                            nbytes - sum(
                                shard_state[
                                    _tq_sparse_expert_key(output_key, eid)
                                ].numel()
                                * shard_state[
                                    _tq_sparse_expert_key(output_key, eid)
                                ].element_size()
                                for eid in missing
                            ),
                        )
                        continue

                    # Normal dense residual tensor.
                    shard_state[output_key] = tensor.contiguous()
                    residual_bytes += nbytes

            if total_shards == 1:
                out_name = "model.safetensors"
            else:
                out_name = (
                    f"model-{shard_i:05d}-of-{total_shards:05d}.safetensors"
                )

            final_path = tq_dir / out_name
            part_path = tq_dir / (out_name + ".part")

            if part_path.exists():
                part_path.unlink()

            # Only one extra residual shard exists on disk at a time.
            save_file(
                shard_state,
                str(part_path),
                metadata={"format": "pt"},
            )
            part_path.replace(final_path)

            for saved_key in shard_state:
                weight_map[saved_key] = out_name

            shard_gib = final_path.stat().st_size / (1024 ** 3)
            print(
                f"[{shard_i}/{total_shards}] wrote {out_name} "
                f"({shard_gib:.3f} GiB), "
                f"sparse_missing_experts_total={sparse_expert_count}"
            )

            del shard_state

        not_found = replaced_keys - seen_replaced
        if not_found:
            print()
            print(
                "WARNING: these candidate TQ source keys were not present "
                "in the base checkpoint:"
            )
            for key in sorted(not_found):
                print("  ", key)

        partial_not_found = (
            set(partially_covered_fused) - seen_partial
        )
        if partial_not_found:
            print()
            print(
                "WARNING: these partial fused MoE sources were not present "
                "in the base checkpoint:"
            )
            for key in sorted(partial_not_found):
                print("  ", key)

        if total_shards > 1:
            index = {
                "metadata": {"total_size": int(residual_bytes)},
                "weight_map": weight_map,
            }
            _write_json_atomic(
                tq_dir / "model.safetensors.index.json",
                index,
            )

        marker.unlink(missing_ok=True)

    except Exception:
        # Keep completed shards so the user can inspect disk usage, but make it
        # explicit that this model directory is not safe to serve.
        print()
        print(
            "Residual reconstruction failed. Completed shards were kept to "
            "avoid wasting work, but .tq_residual_in_progress remains present."
        )
        print(
            "Re-running this script will remove/rebuild the residual shards."
        )
        raise

    return {
        "original_bytes": original_bytes,
        "residual_bytes": residual_bytes,
        "removed_bytes": removed_bytes,
        "dropped_bytes": dropped_bytes,
        "placeholder_count": placeholder_count,
        "seen_replaced": seen_replaced,
        "sparse_expert_count": sparse_expert_count,
        "sparse_saved_bytes": sparse_saved_bytes,
        "sparse_original_bytes": sparse_original_bytes,
    }



def _tq_json_safe(x):
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): _tq_json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, torch.Size)):
        return [_tq_json_safe(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, torch.dtype):
        return str(x)
    return str(x)


def _tq_make_runtime_packed(tensors: dict) -> torch.Tensor:
    """Return the final runtime packed payload for old and new TQ records.

    New native-core / n-gram records already store ``packed_all`` directly.
    Older records store the five historical packed components and are assembled
    into the same [..., 520] runtime layout here.
    """
    packed_all = tensors.get("packed_all")
    if isinstance(packed_all, torch.Tensor):
        packed = packed_all.detach().cpu().to(torch.uint8).contiguous()
        if packed.ndim not in (2, 3) or int(packed.shape[-1]) != 520:
            raise RuntimeError(
                f"Unexpected direct TQ packed_all shape {tuple(packed.shape)}; "
                "expected [B,520] or [chunks,B,520]"
            )
        return packed

    required = (
        "SigRec1_select_packed",
        "SigRec2_select_packed",
        "SigRec3_select_packed",
        "SigRec4_select_packed",
        "X567_packed",
    )
    missing = [key for key in required if key not in tensors]
    if missing:
        raise RuntimeError(
            "TQ record contains neither direct packed_all nor the complete "
            f"legacy packed components; missing={missing}, keys={sorted(tensors)}"
        )

    sig1 = tensors["SigRec1_select_packed"].detach().cpu().contiguous()
    sig2 = tensors["SigRec2_select_packed"].detach().cpu().contiguous()
    sig3 = tensors["SigRec3_select_packed"].detach().cpu().contiguous()
    sig4 = tensors["SigRec4_select_packed"].detach().cpu().contiguous()
    x567 = tensors["X567_packed"].detach().cpu().contiguous()
    B = int(sig1.shape[0])
    if (
        tuple(sig1.shape) != (B, 7)
        or tuple(sig2.shape) != (B, 12)
        or tuple(sig3.shape) != (B, 29)
        or tuple(sig4.shape) != (B, 83)
        or tuple(x567.shape) != (3 * B, 128)
    ):
        raise RuntimeError("Unexpected legacy TQ packed record shape")

    packed = torch.zeros((B, 520), dtype=torch.uint8)
    packed[:, 0:7] = sig1.to(torch.uint8)
    packed[:, 8:20] = sig2.to(torch.uint8)
    packed[:, 20:49] = sig3.to(torch.uint8)
    packed[:, 52:135] = sig4.to(torch.uint8)
    packed[:, 136:264] = x567[:B].to(torch.uint8)
    packed[:, 264:392] = x567[B:2 * B].to(torch.uint8)
    packed[:, 392:520] = x567[2 * B:3 * B].to(torch.uint8)
    return packed.contiguous()


def _tq_runtime_scale_tensor(tensors: dict, key: str) -> torch.Tensor:
    """Normalize u_W/std_W without collapsing n-gram per-chunk vectors."""
    value = tensors.get(key)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"TQ record is missing tensor {key!r}")
    value = value.detach().cpu().to(torch.float32).contiguous()
    if value.ndim == 0:
        return value.reshape(1)
    return value


def _tq_index_name_candidates(
    layer_name: str,
    *,
    multimodal: bool,
) -> list[str]:
    """Names that may represent the same record before/after runtime rewrite."""
    name = str(layer_name)
    out = [name]

    def add(x: str):
        if x and x not in out:
            out.append(x)

    add(tq_layer_name_to_runtime(name, multimodal=multimodal))

    short_mm = "model.language_model."
    full_mm = "model.language_model.model."

    if name.startswith(full_mm):
        rest = name[len(full_mm):]
        add(short_mm + rest)
        add("model." + rest)

    if name.startswith(short_mm) and not name.startswith(full_mm):
        rest = name[len(short_mm):]
        if rest.startswith(("layers.", "embed_tokens.", "norm.")):
            add(full_mm + rest)
            add("model." + rest)

    if name.startswith("model.") and not name.startswith("model.language_model."):
        rest = name[len("model."):]
        add(full_mm + rest)
        add(short_mm + rest)

    if name == "lm_head" or name.startswith("lm_head."):
        add("model.language_model." + name)

    if name.startswith("model.language_model.lm_head"):
        add(name[len("model.language_model."):])

    return out


def _find_existing_index_record(
    locations: dict,
    layer_name: str,
    *,
    multimodal: bool,
) -> tuple[str | None, dict | None]:
    for candidate in _tq_index_name_candidates(
        layer_name,
        multimodal=multimodal,
    ):
        loc = locations.get(candidate)
        if isinstance(loc, dict):
            return candidate, loc
    return None, None


def _find_local_pt_record(
    quant_dir: Path,
    layer_name: str,
    *,
    multimodal: bool,
) -> tuple[str | None, Path | None]:
    for candidate in _tq_index_name_candidates(
        layer_name,
        multimodal=multimodal,
    ):
        p = quant_dir / f"{sanitize_layer_name(candidate)}.pt"
        if p.exists():
            return candidate, p
    return None, None


def _next_incremental_generation(quant_dir: Path) -> int:
    pat = re.compile(r"^tq-inc-(\d{4})-\d{5}-of-\d{5}\.safetensors$")
    best = 0
    for p in quant_dir.glob("tq-inc-*.safetensors"):
        m = pat.match(p.name)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def _preflight_torch_record_file(path: Path) -> None:
    """Cheap structural validation for one per-layer torch.save record.

    We avoid deserializing all tensors twice. Modern torch.save files are ZIP
    archives; very old/legacy files may be TAR archives; some historical
    PyTorch formats are raw pickle streams. For ZIP/TAR we validate the
    container structure here. Unknown/raw-pickle formats are left for the real
    torch.load() call.
    """
    st_before = path.stat()
    if st_before.st_size <= 0:
        raise RuntimeError("empty .pt file")

    with open(path, "rb") as f:
        magic = f.read(4)

    if magic.startswith(b"PK"):
        try:
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                if not any(name.endswith("data.pkl") for name in names):
                    raise RuntimeError("torch ZIP archive lacks data.pkl")
                bad_member = zf.testzip()
                if bad_member is not None:
                    raise RuntimeError(
                        f"torch ZIP archive has corrupt member {bad_member!r}"
                    )
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"invalid torch ZIP archive: {e}") from e
    elif tarfile.is_tarfile(path):
        try:
            with tarfile.open(path, "r:*") as tf:
                names = set(tf.getnames())
                required = {"storages", "tensors", "pickle"}
                missing = sorted(required - names)
                if missing:
                    raise RuntimeError(
                        "legacy torch TAR archive is incomplete; missing "
                        + ", ".join(missing)
                    )
        except (tarfile.TarError, OSError) as e:
            raise RuntimeError(f"invalid legacy torch TAR archive: {e}") from e

    st_after = path.stat()
    if (
        st_after.st_size != st_before.st_size
        or st_after.st_mtime_ns != st_before.st_mtime_ns
    ):
        raise RuntimeError(".pt file changed while converter was validating it")


def _preflight_new_tq_records(
    new_entries: list[tuple[str, str, Path]],
) -> None:
    """Fail before shard creation if any new per-layer .pt record is incomplete."""
    print(
        f"[local-convert] Preflighting {len(new_entries)} .pt record containers..."
    )

    bad = []
    now = time.time()

    for idx, (manifest_name, pt_logical_name, path) in enumerate(new_entries, 1):
        try:
            _preflight_torch_record_file(path)
        except Exception as e:
            try:
                age_s = max(0.0, now - path.stat().st_mtime)
                size = path.stat().st_size
            except OSError:
                age_s = -1.0
                size = -1

            bad.append(
                {
                    "manifest_name": manifest_name,
                    "pt_logical_name": pt_logical_name,
                    "path": str(path),
                    "size": size,
                    "age_seconds": age_s,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

        if idx % 10000 == 0:
            print(f"[local-convert] preflight {idx}/{len(new_entries)}")

    if bad:
        lines = []
        for item in bad[:50]:
            age = item["age_seconds"]
            age_text = "unknown" if age < 0 else f"{age:.1f}s"
            lines.append(
                "  logical=" + item["manifest_name"] + "\n"
                "  file=" + item["path"] + "\n"
                f"  size={item['size']} bytes, age={age_text}\n"
                "  error=" + item["error"]
            )

        raise RuntimeError(
            f"Found {len(bad)} invalid/incomplete TQ .pt record(s) BEFORE "
            "safetensor conversion. No new TQ shard was committed.\n\n"
            + "\n\n".join(lines)
            + "\n\nIf the listed file is very recent, the quantizer was probably still "
              "writing it; finish/stop quantization before building. If it is old, "
              "delete only that corrupt .pt file and rerun layer-by-layer quantization "
              "so the missing record is regenerated."
        )

    print("[local-convert] .pt preflight passed")



def _pt_record_source_keys(meta: dict) -> list[str]:
    raw = meta.get("source_state_keys")
    if isinstance(raw, (list, tuple)) and raw:
        return [str(x) for x in raw if x]

    one = meta.get("source_state_key")
    return [str(one)] if one else []


def _strip_weight_suffix(name: str) -> str:
    name = str(name)
    return name[:-7] if name.endswith(".weight") else name


def _recover_logical_name_from_pt_record(pt_path: Path) -> tuple[str, dict]:
    """Recover one quantizer logical target name from record provenance."""
    rec = torch.load(
        pt_path,
        map_location="cpu",
        weights_only=False,
    )
    try:
        meta = rec.get("meta", {})
        if not isinstance(meta, dict):
            raise RuntimeError("record['meta'] is not a dict")

        for key in ("target_name", "layer_name", "logical_name"):
            value = meta.get(key)
            if isinstance(value, str) and value:
                return value, dict(meta)

        source_keys = _pt_record_source_keys(meta)
        if not source_keys:
            raise RuntimeError(
                "cannot recover logical name: missing source_state_key(s)"
            )

        source_base = _strip_weight_suffix(source_keys[0])
        kind = str(meta.get("target_kind") or "")
        slice_index = meta.get("slice_index")

        # Final Qwen4Exp physical n-gram record.
        if kind in {"tq_embedding_shard", "tq_ngram_chunked_shard"}:
            return source_base, dict(meta)

        # Backward compatibility with older artificial chunk format.
        if kind == "tq_embedding_chunk":
            row_start = meta.get("row_start")
            row_end = meta.get("row_end")
            if row_start is None or row_end is None:
                raise RuntimeError(
                    "tq_embedding_chunk missing row_start/row_end"
                )
            name = (
                f"{source_base}.tq_embedding_chunk_"
                f"{int(row_start):012d}_{int(row_end):012d}"
            )
            return name, dict(meta)

        # Native-vLLM fused names.
        fused_prefix = "native_vllm_fused_"
        if kind.startswith(fused_prefix):
            component = kind[len(fused_prefix):]
            expert = False
            if component.startswith("expert_"):
                expert = True
                component = component[len("expert_"):]

            if "." not in source_base:
                raise RuntimeError(
                    f"cannot infer fused target prefix from {source_base!r}"
                )

            prefix = source_base.rsplit(".", 1)[0]
            name = f"{prefix}.{component}"

            if expert or slice_index is not None:
                if slice_index is None:
                    raise RuntimeError(
                        f"{kind!r} is expert-fused but slice_index is missing"
                    )
                name += f".expert_{int(slice_index):04d}"

            return name, dict(meta)

        # Ordinary/expert checkpoint target.
        name = source_base
        if slice_index is not None:
            name += f".expert_{int(slice_index):04d}"

        return name, dict(meta)

    finally:
        del rec


def _augment_manifest_from_leftover_pt_records(
    tq_dir: Path,
    *,
    multimodal: bool,
) -> dict:
    """Merge valid leftover per-target .pt records into _layers.json."""
    quant_dir = tq_dir / QUANT_DIR_NAME
    layers_path = quant_dir / "_layers.json"

    with open(layers_path, "r", encoding="utf-8") as f:
        layers = [str(x) for x in json.load(f)]

    existing = set(layers)
    pt_files = sorted(quant_dir.glob("*.pt"))

    if not pt_files:
        return {
            "pt_files": 0,
            "added": 0,
            "already_manifested": 0,
            "bad": [],
        }

    print(
        f"[manifest-recovery] inspecting {len(pt_files)} leftover .pt records...",
        flush=True,
    )

    added = []
    already = 0
    bad = []

    for i, pt_path in enumerate(pt_files, 1):
        if i == 1 or i % 1000 == 0 or i == len(pt_files):
            print(
                f"[manifest-recovery] {i}/{len(pt_files)}",
                flush=True,
            )

        try:
            logical_name, _meta = _recover_logical_name_from_pt_record(pt_path)
        except Exception as exc:
            bad.append((pt_path, exc))
            continue

        # Verify metadata-derived logical name maps back to this physical file.
        candidates = _tq_index_name_candidates(
            logical_name,
            multimodal=multimodal,
        )
        candidate_files = {
            quant_dir / f"{sanitize_layer_name(x)}.pt"
            for x in candidates
        }
        if pt_path not in candidate_files:
            bad.append(
                (
                    pt_path,
                    RuntimeError(
                        "metadata-derived logical name does not map back to "
                        f"this filename: logical={logical_name!r}"
                    ),
                )
            )
            continue

        if logical_name in existing:
            already += 1
            continue

        layers.append(logical_name)
        existing.add(logical_name)
        added.append(logical_name)

    if bad:
        preview = "\n".join(
            f"  file={p}\n"
            f"  error={type(exc).__name__}: {exc}"
            for p, exc in bad[:50]
        )
        raise RuntimeError(
            f"Found {len(bad)} leftover .pt record(s) whose logical target "
            "could not be recovered safely. Refusing to delete or ignore them.\n"
            + preview
        )

    if added:
        tmp = layers_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(layers, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(layers_path)

        print(
            f"[manifest-recovery] added {len(added)} valid leftover .pt "
            f"records to _layers.json ({len(layers)} total)",
            flush=True,
        )
        for name in added[:20]:
            print(f"  + {name}")
        if len(added) > 20:
            print(f"  ... and {len(added) - 20} more")
    else:
        print(
            "[manifest-recovery] no missing manifest entries found",
            flush=True,
        )

    return {
        "pt_files": len(pt_files),
        "added": len(added),
        "already_manifested": already,
        "bad": [],
    }


def convert_local_tq_pt_to_safetensors(
    tq_dir: Path,
    repo_id: str,
    delete_pt_after_success: bool = True,
) -> dict:
    """Incrementally merge newly quantized .pt records into local TQ shards.

    Existing tq-*.safetensors and their tq_shards.json entries are preserved.
    Only manifest records that are NOT already represented by the existing
    safetensor index are read from .pt and packed into new incremental shards.

    Transactional behavior:
      1. validate every existing indexed shard;
      2. identify only genuinely new manifest records;
      3. build new safetensor shard(s) in a temp directory;
      4. move new shard(s) into quantization_data/;
      5. atomically replace tq_shards.json with the merged index;
      6. only then delete the newly converted .pt files.

    Therefore a rerun after adding 7,935 new quantized records does not repack
    or rewrite the tens of thousands of records already stored in safetensors.
    """
    quant_dir = tq_dir / QUANT_DIR_NAME
    multimodal = detect_multimodal_target(tq_dir)

    # _layers.json can lag behind successfully written per-target .pt files
    # after an interrupted/resumed huge-model run. Recover those records first.
    recovery = _augment_manifest_from_leftover_pt_records(
        tq_dir,
        multimodal=multimodal,
    )

    layers = [str(x) for x in load_tq_manifest(tq_dir)]
    if recovery.get("added", 0):
        print(
            f"[local-convert] manifest recovered +{recovery['added']} "
            f"record(s); authoritative manifest now {len(layers)}",
            flush=True,
        )

    existing_path = quant_dir / TQ_SHARD_INDEX_NAME

    index = {
        "format": "tq_safetensors_shards_v1",
        "repo_id": repo_id,
        "quant_dir": QUANT_DIR_NAME,
        "target_shard_mb": TQ_SHARD_TARGET_MB,
        "num_shards": 0,
        "layers": {},
    }

    source_index_path = existing_path if existing_path.exists() else None

    if source_index_path is not None:
        with open(source_index_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        if not isinstance(loaded.get("layers"), dict):
            raise RuntimeError(
                f"{source_index_path} does not contain a dict at key 'layers'"
            )

        index.update(loaded)
        index["format"] = "tq_safetensors_shards_v1"
        index["repo_id"] = repo_id
        index["quant_dir"] = QUANT_DIR_NAME
        index["target_shard_mb"] = TQ_SHARD_TARGET_MB

    locations = index["layers"]

    # Validate all existing shard references before modifying anything.
    existing_shard_names = sorted({
        Path(str(v["filename"])).name
        for v in locations.values()
        if isinstance(v, dict) and v.get("filename")
    })
    missing_existing_shards = [
        name
        for name in existing_shard_names
        if not (quant_dir / name).exists()
    ]
    if missing_existing_shards:
        raise RuntimeError(
            "Existing tq_shards.json references missing local safetensors:\n  "
            + "\n  ".join(missing_existing_shards[:100])
        )

    new_entries: list[tuple[str, str, Path]] = []
    already_indexed = 0
    unresolved = []

    for logical_name in layers:
        matched_name, _ = _find_existing_index_record(
            locations,
            logical_name,
            multimodal=multimodal,
        )
        if matched_name is not None:
            already_indexed += 1
            continue

        pt_logical_name, pt_path = _find_local_pt_record(
            quant_dir,
            logical_name,
            multimodal=multimodal,
        )
        if pt_path is None:
            unresolved.append(logical_name)
            continue

        new_entries.append((logical_name, pt_logical_name, pt_path))

    if unresolved:
        raise RuntimeError(
            f"{len(unresolved)} manifest records are neither already present "
            "in tq_shards.json nor available as local .pt records.\n"
            "First unresolved records:\n  "
            + "\n  ".join(unresolved[:100])
        )

    print()
    print("[local-convert] Incremental TQ merge")
    print(f"[local-convert] Manifest records: {len(layers)}")
    print(f"[local-convert] Already in safetensors: {already_indexed}")
    print(f"[local-convert] New .pt records to convert: {len(new_entries)}")
    print(f"[local-convert] Existing TQ shard files: {len(existing_shard_names)}")

    if new_entries:
        _preflight_new_tq_records(new_entries)

    if not new_entries:
        index["num_shards"] = len(existing_shard_names)


        print("[local-convert] Nothing new to convert.")
        return {
            "num_shards": len(existing_shard_names),
            "num_layers": len(locations),
            "new_layers": 0,
            "deleted_pt": 0,
            "already_converted": True,
        }

    # Pack ONLY the new records by source .pt size.
    groups = []
    cur = []
    cur_bytes = 0

    for item in new_entries:
        size = item[2].stat().st_size
        if cur and cur_bytes + size > TQ_SHARD_TARGET_BYTES:
            groups.append(cur)
            cur = []
            cur_bytes = 0
        cur.append(item)
        cur_bytes += size

    if cur:
        groups.append(cur)

    generation = _next_incremental_generation(quant_dir)

    tmp = quant_dir / ".tq_local_convert_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    merged_locations = dict(locations)
    generated_names = []
    converted_pt_paths = []
    record_counter = 0

    try:
        for i, group in enumerate(groups, 1):
            shard_name = (
                f"tq-inc-{generation:04d}-"
                f"{i:05d}-of-{len(groups):05d}.safetensors"
            )
            rel = f"{QUANT_DIR_NAME}/{shard_name}"
            tensors_out = {}

            print(
                f"[local-convert] new shard {i}/{len(groups)}: "
                f"{len(group)} records -> {shard_name}"
            )

            for manifest_name, pt_logical_name, src in group:
                rec = torch.load(
                    src,
                    map_location="cpu",
                    weights_only=False,
                )
                tensors = rec.get("tensors", {})
                meta = rec.get("meta", {})

                rid = f"r{record_counter:08d}"
                record_counter += 1

                packed_key = f"{rid}.packed_all"
                u_key = f"{rid}.u_W"
                std_key = f"{rid}.std_W"

                tensors_out[packed_key] = _tq_make_runtime_packed(tensors)
                tensors_out[u_key] = _tq_runtime_scale_tensor(tensors, "u_W")
                tensors_out[std_key] = _tq_runtime_scale_tensor(tensors, "std_W")

                # Keep the manifest logical name here. The normal runtime-name
                # rewrite runs immediately after this conversion and rewrites
                # BOTH _layers.json and tq_shards.json together.
                merged_locations[manifest_name] = {
                    "filename": rel,
                    "meta": _tq_json_safe(meta),
                    "tensor_keys": {
                        "packed_all": packed_key,
                        "u_W": u_key,
                        "std_W": std_key,
                    },
                }

                converted_pt_paths.append(src)
                del rec

            tmp_shard = tmp / shard_name
            save_file(
                tensors_out,
                str(tmp_shard),
                metadata={"format": "tq_safetensors_shard_v1"},
            )
            generated_names.append(shard_name)
            del tensors_out

        # Move the newly generated shard files first. Existing shard files are
        # never removed or rewritten.
        for shard_name in generated_names:
            dest = quant_dir / shard_name
            if dest.exists():
                raise RuntimeError(
                    f"Refusing to overwrite existing incremental shard: {dest}"
                )
            shutil.move(str(tmp / shard_name), str(dest))

        merged_index = dict(index)
        merged_index["layers"] = merged_locations
        all_shard_names = {
            Path(str(v["filename"])).name
            for v in merged_locations.values()
            if isinstance(v, dict) and v.get("filename")
        }
        merged_index["num_shards"] = len(all_shard_names)

        # Atomic index replacement: only after every new shard is in place.
        tmp_index = existing_path.with_suffix(".json.tmp")
        with open(tmp_index, "w", encoding="utf-8") as f:
            json.dump(merged_index, f, indent=2, sort_keys=True)
        tmp_index.replace(existing_path)


        deleted = 0
        if delete_pt_after_success:
            seen = set()
            for pt_path in converted_pt_paths:
                key = str(pt_path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                if pt_path.exists():
                    pt_path.unlink()
                    deleted += 1

        print(
            f"[local-convert] merged {len(new_entries)} new records into "
            f"{len(groups)} new safetensor shard(s)"
        )
        print(
            f"[local-convert] total indexed records: "
            f"{len(merged_locations)}"
        )
        print(
            f"[local-convert] total TQ shard files: "
            f"{len(all_shard_names)}"
        )
        print(
            f"[local-convert] deleted {deleted} newly converted .pt files"
        )

        return {
            "num_shards": len(all_shard_names),
            "num_layers": len(merged_locations),
            "new_layers": len(new_entries),
            "new_shards": len(groups),
            "deleted_pt": deleted,
            "already_converted": False,
        }

    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)



def cleanup_orphan_local_pt_files(tq_dir: Path) -> int:
    """Delete leftover legacy .pt records only after validating tq_shards.json.

    The final TQ safetensor index must cover every logical record in
    _layers.json and every referenced safetensor shard must exist locally.
    Once that is true, any remaining quantization_data/*.pt file is legacy or
    stale output and is not required by the final local TQ representation.
    """
    quant_dir = tq_dir / QUANT_DIR_NAME
    layers_path = quant_dir / "_layers.json"
    idx_path = quant_dir / TQ_SHARD_INDEX_NAME

    if not layers_path.exists():
        raise FileNotFoundError(layers_path)
    if not idx_path.exists():
        raise FileNotFoundError(idx_path)

    with open(layers_path, "r", encoding="utf-8") as f:
        manifest_layers = {str(x) for x in json.load(f)}

    with open(idx_path, "r", encoding="utf-8") as f:
        idx = json.load(f)

    indexed_layers = set(map(str, idx.get("layers", {}).keys()))
    missing_from_index = manifest_layers - indexed_layers
    if missing_from_index:
        preview = "\n  ".join(sorted(missing_from_index)[:30])
        raise RuntimeError(
            "Refusing to delete legacy .pt files because tq_shards.json does "
            "not cover the full _layers.json manifest. Missing:\n  " + preview
        )

    shard_names = {
        Path(str(v["filename"])).name
        for v in idx.get("layers", {}).values()
        if isinstance(v, dict) and v.get("filename")
    }
    missing_shards = [
        name for name in sorted(shard_names)
        if not (quant_dir / name).exists()
    ]
    if missing_shards:
        raise RuntimeError(
            "Refusing to delete legacy .pt files because local TQ shard files "
            "are missing:\n  " + "\n  ".join(missing_shards[:30])
        )

    leftovers = sorted(quant_dir.glob("*.pt"))
    if not leftovers:
        print("[local-cleanup] no legacy .pt files remain")
        return 0

    # Never delete a leftover .pt merely because _layers.json forgot it.
    # Require a matching tq_shards.json entry for every physical record.
    multimodal = detect_multimodal_target(tq_dir)
    safe_to_delete = []
    unindexed = []
    unreadable = []

    for p in leftovers:
        try:
            logical_name, _meta = _recover_logical_name_from_pt_record(p)
        except Exception as exc:
            unreadable.append((p, exc))
            continue

        matched_name, loc = _find_existing_index_record(
            idx.get("layers", {}),
            logical_name,
            multimodal=multimodal,
        )

        if matched_name is None or not isinstance(loc, dict):
            unindexed.append((p, logical_name))
        else:
            safe_to_delete.append(p)

    if unreadable or unindexed:
        details = []
        for p, exc in unreadable[:30]:
            details.append(
                f"  unreadable={p}\n"
                f"    {type(exc).__name__}: {exc}"
            )
        for p, logical_name in unindexed[:30]:
            details.append(
                f"  unindexed={p}\n"
                f"    logical={logical_name}"
            )
        raise RuntimeError(
            "Refusing to delete leftover .pt files because some are not "
            "safely represented in tq_shards.json. Run normal conversion "
            "again so they are merged first.\n"
            + "\n".join(details)
        )

    print(
        f"[local-cleanup] validated {len(manifest_layers)} manifest records "
        f"across {len(shard_names)} safetensor shards"
    )
    print(
        f"[local-cleanup] deleting {len(safe_to_delete)} .pt files only "
        "after confirming each has an indexed safetensor record"
    )

    for p in safe_to_delete:
        p.unlink()

    return len(safe_to_delete)


def upload_local_tq_model(
    tq_dir: Path,
    repo_id: str,
    hf_token: str | None,
    *,
    num_workers: int = 1,
) -> None:
    """Memory-safe/restartable upload of an already-built local TQ directory."""
    idx_path = tq_dir / QUANT_DIR_NAME / TQ_SHARD_INDEX_NAME
    if not idx_path.exists():
        raise FileNotFoundError(idx_path)

    # Safe now because cleanup_orphan_local_pt_files validates the complete
    # safetensor representation before removing anything.
    deleted_pt = cleanup_orphan_local_pt_files(tq_dir)
    if deleted_pt:
        print(f"[upload] removed {deleted_pt} leftover local .pt files")

    marker = tq_dir / ".tq_residual_in_progress"
    if marker.exists():
        raise RuntimeError(
            f"Refusing to upload an incomplete residual checkpoint: {marker}"
        )

    with open(idx_path, "r", encoding="utf-8") as f:
        idx = json.load(f)

    shard_names = sorted({
        Path(str(v["filename"])).name
        for v in idx.get("layers", {}).values()
        if isinstance(v, dict) and v.get("filename")
    })

    for name in shard_names:
        p = tq_dir / QUANT_DIR_NAME / name
        if not p.exists():
            raise FileNotFoundError(p)

    # Remove tiny legacy index files locally as well.
    if hf_token:
        login(token=hf_token)

    api = HfApi(token=hf_token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=False,
        exist_ok=True,
    )

    print()
    print(f"[upload] repo: {repo_id}")
    print(f"[upload] local directory: {tq_dir}")
    print(f"[upload] TQ shards: {len(shard_names)}")
    print(f"[upload] workers: {num_workers}")
    print("[upload] resumable large-folder mode")
    print("[upload] rerun --upload-only if interrupted")

    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(tq_dir),
        num_workers=max(1, int(num_workers)),
        print_report=True,
        print_report_every=60,
        # Defense-in-depth: these should already be absent locally, but never
        # publish legacy quantizer artifacts if one appears during the run.
        ignore_patterns=[
            "**/*.pt",
                        ".tq_residual_in_progress",
            "quantization_data/.tq_local_convert_tmp/**",
        ],
    )

    print(f"[upload] done: {repo_id}")

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally merge new TQ quantization into safetensors, "
            "rebuild the residual from complete coverage, and upload."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--repo-id",
        required=True,
        help=(
            "Destination Hugging Face repo ID, e.g. username/model-TQ-4bit. "
            "This is used for TQ metadata and upload; no namespace is assumed."
        ),
    )
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--keep-pt", action="store_true")
    parser.add_argument("--skip-residual", action="store_true")
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help=(
            "Do not merge .pt records. Use after a previous run already "
            "successfully created the incremental TQ safetensor shards/index."
        ),
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help=(
            "Only validate/clean and upload the existing local TQ directory. "
            "Do not merge new .pt records or rebuild residuals."
        ),
    )
    parser.add_argument(
        "--upload-workers",
        type=int,
        default=int(os.environ.get("TQ_UPLOAD_WORKERS", "1")),
        help="Upload workers; default 1 to minimize RAM.",
    )
    args = parser.parse_args()

    if args.upload_only and args.no_upload:
        parser.error("--upload-only cannot be combined with --no-upload")
    if args.upload_workers < 1:
        parser.error("--upload-workers must be >= 1")

    tq_dir = Path(
        f"/mnt/d/quantized_models/{args.model.rstrip('/').split('/')[-1]}-TQ-4bit"
    ).resolve()
    if not tq_dir.exists():
        raise FileNotFoundError(tq_dir)

    repo_id = args.repo_id.strip()
    if not repo_id or "/" not in repo_id:
        parser.error(
            "--repo-id must be a Hugging Face repo ID in the form owner/repository"
        )

    if args.upload_only:
        print("UPLOAD-ONLY MODE")
        print("Local TQ directory:", tq_dir)
        print("HF TQ repo:", repo_id)
        upload_local_tq_model(
            tq_dir,
            repo_id,
            os.environ.get("HF_TOKEN"),
            num_workers=args.upload_workers,
        )
        return

    multimodal = detect_multimodal_target(tq_dir)
    model_family = get_tq_model_family(tq_dir)
    print(
        "Target runtime:",
        f"{model_family} ConditionalGeneration / multimodal"
        if multimodal
        else f"{model_family} CausalLM / text-only",
    )
    print("Local TQ directory:", tq_dir)
    print("HF TQ repo:", repo_id)

    # --------------------------------------------------------------
    # STEP 1: merge ONLY newly quantized .pt records into the existing
    # safetensor index. Existing safetensor records/shards are preserved.
    #
    # This MUST happen before runtime-name rewriting because after a
    # re-quantization run _layers.json can contain new records that are
    # legitimately absent from the old tq_shards.json.
    # --------------------------------------------------------------
    if not args.skip_convert:
        conversion = convert_local_tq_pt_to_safetensors(
            tq_dir,
            repo_id,
            delete_pt_after_success=not args.keep_pt,
        )

        print()
        print(
            f"[local] total TQ shards: {conversion['num_shards']} | "
            f"new records merged: {conversion.get('new_layers', 0)} | "
            f"deleted new .pt: {conversion['deleted_pt']}"
        )
    else:
        print(
            "[local-convert] skipped (--skip-convert); using existing "
            "tq_shards.json and TQ safetensor shards"
        )

    # --------------------------------------------------------------
    # STEP 2: now that tq_shards.json contains old + new records,
    # normalize the COMPLETE manifest/index to runtime names.
    # --------------------------------------------------------------
    remap = rewrite_tq_manifest_for_runtime(
        tq_dir,
        multimodal=multimodal,
    )

    layer_names = load_tq_manifest(tq_dir)
    plan = build_replacement_plan(tq_dir, layer_names)

    if remap["changed"]:
        print(
            f"Remapped {len(remap['changed'])} TQ logical names "
            "to runtime names"
        )
    else:
        print("TQ logical record names already match target runtime.")

    print()
    print("TQ records after merge:", len(layer_names))
    print("Whole HF tensors replaced:", len(plan["full_replaced"]))
    print(
        "TQ n-gram physical shard records:",
        plan.get("embedding_shard_records", 0),
    )
    print(
        "Dense n-gram HF shard weights removed from residual:",
        len(plan.get("embedding_shard_sources", set())),
    )
    print(
        "Partially covered fused MoE tensors:",
        len(plan["partially_covered_fused"]),
    )

    # --------------------------------------------------------------
    # STEP 3: rebuild residual from the COMPLETE new coverage.
    #
    # This intentionally rebuilds residual weights because the old
    # residual still contains dense fallbacks for targets that have now
    # been quantized. The quantized safetensor shards themselves are NOT
    # rebuilt.
    # --------------------------------------------------------------
    if not args.skip_residual:
        base_dir = resolve_base_model(args.model)

        write_residual_checkpoint_streaming(
            base_dir=base_dir,
            tq_dir=tq_dir,
            replaced_keys=plan["full_replaced"],
            partially_covered_fused=plan["partially_covered_fused"],
            multimodal=multimodal,
        )
        copy_non_weight_files(base_dir, tq_dir)

    # Final invariant: every manifest entry must now come from safetensors.
    # cleanup_orphan_local_pt_files validates tq_shards.json coverage and
    # referenced shard existence before removing any stale leftover .pt.
    if not args.keep_pt:
        cleanup_orphan_local_pt_files(tq_dir)

    if not args.no_upload:
        upload_local_tq_model(
            tq_dir,
            repo_id,
            os.environ.get("HF_TOKEN"),
            num_workers=args.upload_workers,
        )
    else:
        print("[upload] skipped (--no-upload)")


if __name__ == "__main__":
    main()
