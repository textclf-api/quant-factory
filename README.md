# TQ Quant Factory

**Calibration-free, information-theoretic 4-bit quantization for Large Language Models.**

TQ Quant Factory is the open model-processing pipeline for **TQ**, TextCLF's 4-bit LLM quantization and inference technology.

TQ takes a different approach to post-training quantization: **no calibration dataset and no calibration forward passes are required.**

Instead, TQ approaches quantization from an **information-theoretic, lossy source-coding perspective**, using a proprietary coding-theoretic quantization core designed around the fundamental relationship between **rate and distortion**.

```text
Hugging Face Model
        │
        ▼
   Quant Factory
        │
        │  Calibration-Free
        ▼
   TQ 4-bit Model
        │
        ├────────────► Hugging Face
        │
        ▼
       vLLM
        │
        ▼
 TQ Native Runtime
```

> **Model in. Quantized model out. No calibration dataset required.**

---

## Why TQ?

Many post-training quantization methods rely on representative calibration data.

A typical workflow may involve:

```text
Model
  +
Calibration Dataset
  │
  ▼
Preprocessing
  │
  ▼
Calibration Forward Passes
  │
  ▼
Statistics / Optimization
  │
  ▼
Quantization
  │
  ▼
Quantized Model
```

TQ removes the calibration stage:

```text
Model Weights
     │
     ▼
     TQ
     │
     ▼
Quantized Model
```

**No calibration dataset.**

**No calibration forward passes.**

This makes quantization easier to automate and removes the need to choose a dataset intended to represent future inference workloads.

Calibration-free quantization is particularly useful when:

- representative calibration data is unavailable;
- deployment workloads are not known in advance;
- model or application data is restricted;
- many models need to be quantized automatically;
- reproducible model transformation is important; or
- calibration compute and infrastructure are undesirable.

---

# Quantization Through Information Theory

TQ approaches model quantization as a **lossy source-coding problem**.

Any lossy compression system involves a fundamental tradeoff between:

- **Rate** — how many bits are used to represent information.
- **Distortion** — how much information is lost through compression.

Information theory describes the fundamental relationship between these quantities through the **rate–distortion function**:

```text
R(D)
```

where `R(D)` describes the theoretical minimum rate required to represent a source while maintaining a specified distortion level `D`.

TQ's proprietary quantization core is grounded in **coding-theoretic techniques designed around these fundamental rate–distortion limits**.

Instead of relying on representative activation data to determine how model weights should be represented, TQ treats the weights themselves as the source being compressed.

Conceptually:

```text
                  Model Weights
                       │
                       ▼
              ┌─────────────────┐
              │   TQ Quantizer  │
              │                 │
              │  Source Coding  │
              │  Rate ↔ Dist.   │
              └────────┬────────┘
                       │
                       ▼
              Low-Bit Representation
```

The objective is simple:

> **Represent model weights using fewer bits while minimizing the distortion introduced by that representation.**

The production quantization algorithm, coding construction, internal representation, and inference implementation remain proprietary.

---

# Quant Factory

Quantizing one model manually and building infrastructure capable of quantizing many models are different engineering problems.

**Quant Factory is designed for the second.**

It provides the model-processing and orchestration layer around TQ for transforming pretrained Hugging Face models into quantized, inference-ready artifacts.

Quant Factory can:

- Quantize Hugging Face models to TQ
- Quantize without calibration datasets
- Quantize without calibration forward passes
- Stream model weights incrementally
- Process large models layer by layer
- Persist quantization results incrementally
- Resume interrupted quantization jobs
- Convert intermediate records into final TQ models
- Optionally publish finished models to Hugging Face
- Serve TQ models through vLLM

TQ uses a **streaming, layer-by-layer quantization pipeline**, allowing large models to be processed without loading the entire model into GPU memory at once.

GGUF is not used or required.

---

# Architecture

Quant Factory intentionally separates the **open model-processing pipeline** from the **proprietary quantization and inference technology**.

```text
┌─────────────────────────────────────────────┐
│                Quant Factory                │
│                                             │
│  Hugging Face discovery                    │
│  Model traversal                           │
│  Streaming                                 │
│  Quantization orchestration                │
│  Incremental persistence                   │
│  Resume support                            │
│  Model conversion                          │
│  Hugging Face publishing                   │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
              ┌───────────────────┐
              │  Native TQ Core   │
              │                   │
              │    Proprietary    │
              │    Quantization   │
              └─────────┬─────────┘
                        │
                        ▼
                  TQ 4-bit Model
                        │
                        ▼
                      vLLM
                        │
                        ▼
              ┌───────────────────┐
              │ TQ Native Runtime │
              │                   │
              │    Proprietary    │
              │     Inference     │
              └───────────────────┘
```

## Open Pipeline

The visible code in this repository is responsible for:

- Hugging Face checkpoint discovery
- Model-specific target discovery
- Streaming model weights
- Quantization orchestration
- Incremental persistence
- Resume support
- Model conversion
- Hugging Face integration
- vLLM integration

## Proprietary TQ Core

The production TQ quantization implementation is distributed as a compiled Linux extension:

```text
_tq_core.so
```

The native core contains the proprietary **coding-theoretic TQ quantization algorithm** and executes tensor operations through ATen CUDA.

The Python pipeline passes tensors into the native core but does not contain the proprietary quantization implementation.

This separation keeps the surrounding infrastructure inspectable while protecting the underlying quantization technology.

---

# Quantization

## 1. Quantize a Model

From the TQ quantizer directory:

```bash
./quantize-model meta-llama/Llama-3.1-8B-Instruct
```

**No calibration dataset needs to be supplied.**

TQ streams the source checkpoint incrementally, quantizes supported targets, and persists results as the model is processed.

The default output directory is:

```text
/mnt/d/quantized_models/Llama-3.1-8B-Instruct-TQ-4bit
```

You can specify another location:

```bash
./quantize-model meta-llama/Llama-3.1-8B-Instruct \
  --output-dir /path/to/output
```

Or configure a different output root:

```bash
export TQ_OUTPUT_ROOT=/path/to/quantized_models
```

The quantization process is incremental. Completed records are persisted instead of being held entirely in memory.

---

## 2. Build the Final TQ Model

After quantization finishes:

```bash
python build_convert_upload_tq_model_incremental_qwen4exp.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --repo-id YOUR_USERNAME/Llama-3.1-8B-Instruct-TQ-4bit \
  --no-upload
```

This converts the incremental quantization records into the final TQ model layout expected by the runtime.

`--repo-id` identifies the Hugging Face repository associated with the finished model.

For example:

```bash
--repo-id my-user/Llama-3.1-8B-Instruct-TQ-4bit
```

---

## 3. Build and Upload to Hugging Face

To publish the finished model to Hugging Face, omit `--no-upload`:

```bash
python build_convert_upload_tq_model_incremental_qwen4exp.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --repo-id YOUR_USERNAME/Llama-3.1-8B-Instruct-TQ-4bit
```

You must be authenticated with Hugging Face and have permission to create or update the specified repository.

---

# Inference

TQ models can be served through the TextCLF TQ integration for **vLLM**.

The inference integration consists of:

- visible Python integration code for vLLM; and
- compiled native extensions implementing the proprietary TQ runtime.

The native runtime contains the optimized TQ operators and CUDA execution required to run TQ models efficiently.

The **quantization core and inference core are both proprietary**.

GGUF conversion is not required.

---

## Install

From the repository:

```bash
cd inference
pip install .
```

---

## Serve a TQ Model

Use a TQ model with vLLM by selecting the TQ quantization backend:

```bash
vllm serve YOUR_USERNAME/Llama-3.1-8B-Instruct-TQ-4bit \
  --quantization tq
```

For example:

```bash
vllm serve textclf/Llama-3.1-8B-Instruct-TQ-4bit \
  --quantization tq
```

vLLM loads the TQ metadata and packed weights and dispatches supported operations through the TQ native runtime.

---

# Complete Workflow

```text
                  Hugging Face Model
                         │
                         ▼
                 ┌───────────────┐
                 │ Quant Factory │
                 └───────┬───────┘
                         │
                  Calibration-Free
                   Layer-by-Layer
                         │
                         ▼
                Incremental TQ Records
                         │
                         ▼
                   Model Conversion
                         │
                         ▼
                    TQ 4-bit Model
                         │
                 ┌───────┴────────┐
                 │                │
                 ▼                ▼
           Hugging Face          vLLM
                                  │
                                  ▼
                           TQ Native Runtime
```

In short:

```bash
# 1. Quantize — no calibration dataset required
./quantize-model meta-llama/Llama-3.1-8B-Instruct

# 2. Build the final TQ model
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

---

# Open Pipeline, Proprietary Core

Quant Factory uses a hybrid open/proprietary architecture.

### Available in this repository

- Quantization orchestration
- Model discovery
- Streaming pipeline
- Incremental persistence
- Resume logic
- Model conversion
- Hugging Face integration
- vLLM integration

### Proprietary TQ Technology

The following components are intentionally not open sourced:

- Core quantization algorithm
- Coding-theoretic construction
- Internal weight representation
- Native quantization implementation
- Optimized inference operators
- Native CUDA runtime

These components are distributed as compiled extensions rather than source code.

---

# Design Principles

## Calibration-Free

TQ does not require a representative calibration dataset.

```text
0 calibration samples
```

## No Calibration Forward Passes

TQ operates directly on model weights without running a calibration corpus through the model.

## Information-Theoretic

TQ approaches quantization as a lossy source-coding problem and is grounded in coding-theoretic techniques designed around fundamental rate–distortion limits.

## Streaming

Models are processed incrementally instead of requiring the complete model to reside in GPU memory.

## Reproducible

Removing the calibration dataset eliminates an external, workload-dependent input from the quantization pipeline.

## Deployment-Oriented

TQ is designed as both a quantization technology and an inference system, with direct integration into vLLM.

---

# Why Calibration-Free Matters

Calibration-based quantization can achieve excellent results.

However, calibration introduces another dependency into the model optimization pipeline.

A calibrated quantization run may depend on:

```text
Model
+
Calibration Dataset
+
Selected Samples
+
Preprocessing
+
Sequence Length
+
Calibration Configuration
```

That can be perfectly reasonable when optimizing an individual model for a known workload.

It becomes more complicated when quantization needs to operate as infrastructure across many models and unknown future workloads.

TQ removes the dataset component:

```text
Model
+
Quantization Configuration
        │
        ▼
     TQ Model
```

There is no calibration corpus to select, download, preprocess, version, or maintain.

There is also no calibration distribution that needs to approximate future model inputs.

This is particularly important for **Quant Factory**.

A quantization factory should be able to transform models automatically and reproducibly without requiring a new representative dataset for every model or workload.

---

# Why Rate–Distortion?

Quantization is ultimately a compression problem.

Given a fixed number of bits, we want to preserve as much of the original information as possible.

Or equivalently:

> Given an acceptable level of distortion, how few bits can we use?

This is exactly the type of problem studied by **rate–distortion theory**.

```text
          Lower Distortion
                ▲
                │
                │
                │
                │
                └──────────────────► Lower Rate
                     Compression
```

TQ brings this information-theoretic perspective to LLM weight quantization.

Rather than treating quantization purely as a rounding problem, TQ treats model weights as a source that must be efficiently represented under a constrained bit budget.

The specific coding construction and production algorithm used to accomplish this remain part of the proprietary TQ core.

---

# TQ vs. Calibration-Based Quantization

|                                | TQ                  | Calibration-Based PTQ |
| ------------------------------ | ------------------- | --------------------- |
| Calibration dataset            | **Not required**    | Typically required    |
| Calibration forward passes     | **Not required**    | Typically required    |
| Representative workload needed | **No**              | Often                 |
| Dataset preprocessing          | **None**            | Typically required    |
| Dataset versioning             | **None**            | May be required       |
| Weight streaming               | **Yes**             | Method-dependent      |
| Automated model processing     | **Designed for it** | Method-dependent      |
| vLLM inference                 | **Yes**             | Method-dependent      |

Calibration-free does **not** mean that calibration-based approaches are inherently inferior.

Calibration gives a quantizer additional information about model behavior and can be valuable for aggressive low-bit compression.

TQ explores a different engineering and theoretical tradeoff:

> **Can high-quality low-bit LLM quantization be achieved without making representative data part of the quantization process?**

---

# TQ Philosophy

We believe quantization should eventually behave like infrastructure.

You should not need to build a new calibration pipeline every time you want to optimize another model.

The desired workflow is:

```text
Pretrained Model
       │
       ▼
    Quantize
       │
       ▼
     Deploy
       │
       ▼
      Infer
```

That's what we're building with TQ and Quant Factory.

**Quantize. Deploy. Infer.**

---

# Learn More

For more information about TQ, calibration-free quantization, benchmarks, supported models, and TextCLF:

**[https://www.textclf.com](https://www.textclf.com)**

Technical background:

**[https://www.textclf.com/blog/calibration-free-quantization](https://www.textclf.com/blog/calibration-free-quantization)**

GitHub:

**[https://github.com/textclf-api/quant-factory](https://github.com/textclf-api/quant-factory)**

---

# About TextCLF

TextCLF is building infrastructure for efficient Large Language Models.

TQ combines **calibration-free quantization**, an **information-theoretic approach to model compression**, and a **purpose-built inference runtime** to make large models smaller and more efficient to deploy.

Quant Factory provides the open pipeline around that technology for transforming pretrained models into TQ-quantized, inference-ready artifacts.

**Quantize. Deploy. Infer.**
