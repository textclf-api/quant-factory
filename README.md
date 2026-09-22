# TQ Quantizer

TQ is a 4-bit LLM quantization format designed for efficient inference with vLLM.

This repository provides the tooling required to:

* Quantize Hugging Face models to TQ
* Convert the quantized output into the TQ runtime format
* Optionally publish the finished model to Hugging Face
* Serve TQ models directly with vLLM

TQ uses a **streaming, layer-by-layer quantization pipeline**, allowing large models to be quantized without loading the entire model into GPU memory at once.

GGUF is not used or supported.

---

## Architecture

The quantization pipeline is intentionally split between a lightweight Python layer and a compiled native core.

### Python layer

The visible Python code is responsible for:

* Hugging Face checkpoint discovery
* Streaming model weights one target at a time
* Model-specific target discovery
* Incremental quantization
* Resume support
* Incremental persistence
* Conversion to the final TQ model layout
* Optional Hugging Face upload

### Native TQ core

The proprietary quantization implementation is distributed as the compiled Linux extension:

```text
_tq_core.so
```

The native core contains the private TQ quantization algorithms, including the SC implementation, and executes tensor operations through ATen CUDA.

The Python quantizer passes tensors into the native core but does not contain the proprietary quantization implementation.

---

# Quantization

## 1. Quantize a model

From the TQ quantizer directory:

```bash
./quantize-model meta-llama/Llama-3.1-8B-Instruct
```

TQ streams the source checkpoint layer by layer, quantizes supported targets, and saves the results incrementally.

The default output directory is:

```text
/mnt/d/quantized_models/Llama-3.1-8B-Instruct-TQ-4bit
```

You can specify another location with:

```bash
./quantize-model meta-llama/Llama-3.1-8B-Instruct \
  --output-dir /path/to/output
```

or set a different output root:

```bash
export TQ_OUTPUT_ROOT=/path/to/quantized_models
```

The quantization process is incremental, so completed records are persisted as the model is processed rather than being held entirely in memory.

---

## 2. Build the final TQ model

After quantization finishes, run the converter:

```bash
python build_convert_upload_tq_model_incremental_qwen4exp.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --repo-id YOUR_USERNAME/Llama-3.1-8B-Instruct-TQ-4bit \
  --no-upload
```

This converts the incremental quantization records into the final TQ model layout expected by the runtime.

`--repo-id` identifies the Hugging Face repository that will contain the finished model.

For example:

```bash
--repo-id my-user/Llama-3.1-8B-Instruct-TQ-4bit
```

### Build and upload

To publish the finished model to Hugging Face, omit `--no-upload`:

```bash
python build_convert_upload_tq_model_incremental_qwen4exp.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --repo-id YOUR_USERNAME/Llama-3.1-8B-Instruct-TQ-4bit
```

You must be authenticated with Hugging Face and have permission to create or update the specified repository.

---

# Inference

TQ models are served through the TextCLF TQ integration for vLLM.

The integration consists of two parts:

* Visible Python integration code for vLLM
* Compiled native Linux extensions containing the proprietary TQ runtime implementation

The native runtime handles the sensitive TQ operator implementation and authenticated CUDA execution.

GGUF conversion is not required and is not supported.

## Install

From the repository:

```bash
cd inference
pip install .
```

## Serve a TQ model

Use the model like a normal vLLM model while selecting the TQ quantization backend:

```bash
vllm serve YOUR_USERNAME/Llama-3.1-8B-Instruct-TQ-4bit \
  --quantization tq
```

For example:

```bash
vllm serve textclf/Llama-3.1-8B-Instruct-TQ-4bit \
  --quantization tq
```

vLLM loads the TQ metadata and packed weights and dispatches supported operations through the TQ runtime.

---

# Workflow

The complete workflow is:

```text
Hugging Face model
        │
        ▼
  quantize-model
        │
        │  streaming layer-by-layer
        ▼
Incremental TQ records
        │
        ▼
build_convert_upload_tq_model_incremental_qwen4exp.py
        │
        ▼
 Final TQ model
        │
        ├──────────────► Hugging Face
        │
        ▼
      vLLM
        │
        ▼
  TQ native runtime
```

In short:

```bash
# 1. Quantize
./quantize-model meta-llama/Llama-3.1-8B-Instruct

# 2. Build the final model
python build_convert_upload_tq_model_incremental_qwen4exp.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --repo-id YOUR_USERNAME/Llama-3.1-8B-Instruct-TQ-4bit \
  --no-upload

# 3. Install TQ support for vLLM
cd inference
pip install .

# 4. Serve
vllm serve YOUR_USERNAME/Llama-3.1-8B-Instruct-TQ-4bit \
  --quantization tq
```

## Distribution

The public/customer-facing package contains the Python orchestration and integration layers together with compiled TQ native extensions.

The proprietary TQ quantization and runtime implementations are not distributed as source code.
