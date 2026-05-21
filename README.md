# Telecom Audio LLM Benchmark Pipeline

A rigorous multi-model benchmark comparing next-generation audio-native LLMs against traditional cascade pipelines on a telecom voice-assistant task, backed by a **1M–10M record FAISS customer database**.

---

## Architecture

```text
Audio Input
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Model (Qwen3-Omni / Nemotron-Omni / Whisper+Qwen2.5)           │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌────────────┐  │
│  │  Audio   │   │  TTFT /  │   │   FAISS   │   │  Decode /  │  │
│  │ Encoding │ → │ Planning │ → │ Retrieval │ → │ Synthesis  │  │
│  │  (ASR)   │   │          │   │  (1M–10M) │   │            │  │
│  └──────────┘   └──────────┘   └───────────┘   └────────────┘  │
└──────────────────────────────────────────────────────────────────┘
    │                                   │
    ▼                                   ▼
 Direct Response               Grounded Response
                                       │
                                       ▼
                              Hallucination Judge
                           (BERTScore / LLM-as-Judge)

```

## Files

| File | Purpose |
| --- | --- |
| `generate_customer_db.py` | Generates synthetic 1M–10M customer DB + FAISS IVFFlat index |
| `retrieval_engine.py` | CustomerDB logic, AccuracyTracker, and ToolExecutor |
| `hallucination_judge.py` | BERTScore + LLM-as-Judge hallucination scorer |
| `benchmark_runner.py` | Multi-model benchmarking with granular latency tracing |
| `visualize_results.py` | *(Optional)* Charts + HTML report generator |

## Models Benchmarked

This pipeline relies on heavily customized, dynamic `vLLM` and `transformers` hooks to support bleeding-edge architectures and quantized weights.

| Tag | Model | Architecture | Notes |
| --- | --- | --- | --- |
| `qwen3omni` | `cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit` | Native audio LLM | 4-bit AWQ quantized, dynamically patched |
| `nemotron` | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8` | Native audio LLM | FP8 quantized, Mamba-Transformer hybrid |
| `cascade` | `openai/whisper-large-v3` → `Qwen/Qwen2.5-7B-Instruct` | Cascade ASR+LLM | Traditional non-native baseline |

## Accuracy Metrics

### 1. Tool Selection — Precision / Recall / F1

Measures whether the LLM correctly decides to call `lookup_customer` vs. answer directly.

* **Ground truth**: Inferred from filename (`lookup_<phone>_*.wav` → tool required).

### 2. FAISS Retrieval — Hit-Rate@K

For each tool call where the correct phone number is known:

```text
HR@K = |{samples where true_customer_id ∈ top-K retrieved}| / |total retrieval samples|

```

Reported for K = 1, 3, 5.

### 3. Synthesis Quality — Hallucination Score

Two available backends to ensure the LLM accurately reports database facts:

* **BERTScore** (fast, ~100ms/sample): Embeds final response + flattened customer record with DeBERTa-XL.
* **LLM-as-Judge** (thorough, ~2s/sample): Prompts a small judge model (Qwen2.5-3B) with a strict 0-10 rubic, normalized to [0,1].

## Latency Breakdown

Each sample records 5 isolated latency components, reported as p50 / p75 / p90 / p95 / p99 percentiles:

| Component | Description |
| --- | --- |
| `ASR_ms` | Audio encoding (Native) or Whisper transcription (Cascade) |
| `TTFT_ms` | Time to first token (planning/decision pass) |
| `FAISS_ms` | FAISS IVFFlat search over millions of records |
| `ToolExec_ms` | Full Python tool execution overhead |
| `Decode_ms` | Synthesis / generation pass for final spoken response |

---

## Quick Start (Local Environment)

**1. Generate the customer database**

```bash
# 1 million records (~2 min on CPU)
python generate_customer_db.py --n 1_000_000 --out ./customer_db

```

**2. Run the benchmark**

```bash
python benchmark_runner.py \
    --db ./customer_db \
    --dataset /dataset_generated \
    --models cascade,qwen3omni \
    --judge bertscore \
    --max-samples 500

```

---

## HPC Production Deployment (Singularity/Slurm)

Running complex vLLM instances in HPC environments requires specific cache redirection and path bindings to avoid read-only container crashes and CUDA fork corruption.

Use the following `singularity exec` command template for cluster deployments:

```bash
singularity exec --nv \
  --env HF_TOKEN="your_actual_token_here" \
  --env VLLM_USE_V1=1 \
  --env HF_HOME=/workspace/tmp_cache/hf \
  --env PIP_CACHE_DIR=/workspace/tmp_cache/pip \
  --env XDG_CACHE_HOME=/workspace/tmp_cache \
  --bind /scratch/users/ntu/es0001an/paper:/workspace \
  --bind /home/users/ntu/es0001an/scratch/paper/dataset:/dataset \
  /scratch/users/ntu/es0001an/paper/vllm-omni.sif \
  /workspace/container_venv/bin/python /workspace/benchmark_runner.py \
    --db /workspace/customer_db \
    --dataset /dataset \
    --models qwen3omni \
    --judge bertscore \
    --max-samples 500

```

## Output Structure

```text
./results/
├── comparison_report.json          # Cross-model F1 and Latency summary
├── qwen3omni/
│   ├── traces.csv                  # Per-sample latency + accuracy breakdowns
│   ├── accuracy.json
│   ├── latency.json
│   └── *_synthesis.txt / *_direct.txt
└── cascade/
    └── ...

```

## Key Technical Decisions

* **Dynamic Architecture Injection:** Because Qwen3 Omni models are highly experimental, `benchmark_runner.py` uses a re-entrant `builtins.__import__` monkey-patch to intercept and register custom multimodal configurations (e.g., `_get_num_multimodal_tokens`) into the `transformers` and `vllm` module registries at runtime.
* **HPC Cache Redirection:** Triton JIT, FlashInfer, and HuggingFace caches are forcibly redirected to a `/workspace/tmp_cache/` directory to prevent `PermissionError` crashes when executed inside strictly sandboxed Singularity containers.
* **Balanced Sampling:** Audio datasets are deterministically shuffled using a stable seed prior to limiting via `--max-samples`, ensuring a statistically balanced mix of general-inquiry and tool-lookup audio files are processed during truncated smoke tests.
* **IVFFlat over Flat:** At 1M–10M records, exact `IndexFlatL2` takes 500ms–5s per search. IVFFlat reduces this to 5–60ms while maintaining >95% recall vs. exact search.

