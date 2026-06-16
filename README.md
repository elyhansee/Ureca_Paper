# Telecom Audio LLM Benchmark Pipeline

A rigorous multi-model benchmark comparing next-generation audio-native LLMs against traditional cascade pipelines on a telecom voice-assistant task, backed by a **1M record FAISS customer database**.

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

---

## Repository Structure

| File / Folder | Purpose |
|---|---|
| `generate_customer_db.py` | Generates synthetic 1M customer DB + FAISS IVFFlat index |
| `generate_test.py` | Generates synthetic speech queries (audio waveforms) + labels via local TTS |
| `combine_datasets.py` | Consolidates synthetic tool-use speech waveforms and labels into your main dataset |
| `retrieval_engine.py` | CustomerDB logic, AccuracyTracker, and ToolExecutor |
| `hallucination_judge.py` | BERTScore + LLM-as-Judge hallucination scorer |
| `benchmark_runner.py` | Multi-model benchmarking with granular latency tracing |
| `results/` | Pre-computed benchmark results (`comparison_report.json`) and rendered evaluation charts for the publication |

---

## Models Benchmarked

This pipeline relies on customized, dynamic vLLM and transformers hooks to support bleeding-edge architectures and various precision configurations.

| Tag | Model | Architecture | Notes |
|---|---|---|---|
| `qwen3omni` | `cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit` | Native audio LLM | 4-bit AWQ quantized, dynamically patched |
| `nemotron` | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` | Native audio LLM | BF16 unquantized, Mamba-Transformer hybrid |
| `cascade` | `openai/whisper-large-v3` → `Qwen/Qwen2.5-7B-Instruct` | Cascade ASR+LLM | Traditional non-native baseline |

---

## Results (N=500 samples)

| Model | Mean E2E | Tool F1 | HR@1 | Halluc. Score |
|---|---|---|---|---|
| Cascade (Qwen2.5-7B-Instruct) | **5,529ms** | 0.922 | 0.978 | 0.116 |
| Qwen3-Omni-30B-AWQ | 8,732ms | 0.919 | **1.000** | 0.135 |
| Nemotron-3-Nano-Omni-BF16 | 52,593ms | **0.985** | 0.856 | **0.009** |

Full per-sample traces and latency percentiles available in `results/`.

---

## Accuracy Metrics

### 1. Tool Selection — Precision / Recall / F1

Measures whether the LLM correctly decides to call `lookup_customer` vs. answer directly.

- **Ground truth:** Inferred from filename prefix (`tool_*.wav` → tool required, `general_*.wav` → direct answer)

### 2. FAISS Retrieval — Hit-Rate@K

For each tool call where the correct phone number is known:

```
HR@K = |{samples where true_customer_id ∈ top-K retrieved}| / |total retrieval samples|
```

Reported for K = 1, 3, 5.

### 3. Synthesis Quality — Hallucination Score

Two available backends:

- **BERTScore** (fast, ~100ms/sample): Embeds final response + flattened customer record with DeBERTa-XL, rescaled against the DeBERTa-xlarge-mnli corpus so that 0 = perfect factual alignment and 1 = complete semantic deviation.
- **LLM-as-Judge** (thorough, ~2s/sample): Prompts a small judge model (Qwen2.5-3B) with a strict 0–10 rubric, normalized to [0, 1].

---

## Latency Breakdown

Each sample records 5 isolated latency components, reported as p50 / p75 / p90 / p95 / p99 percentiles:

| Component | Description |
|---|---|
| `ASR_ms` | Audio encoding (native) or Whisper transcription (cascade) |
| `TTFT_ms` | Time to first token (planning/decision pass) |
| `FAISS_ms` | FAISS IVFFlat search over 1M records |
| `ToolExec_ms` | Full Python tool execution overhead |
| `Decode_ms` | Synthesis / generation pass for final response |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install vllm-omni faiss-gpu-cu12 av
```

### 2. Generate the customer database

```bash
# 1 million records (~2 min on GPU)
python generate_customer_db.py --n 1_000_000 --out ./customer_db

# Synthesize the 100 tool-use audio files and tool_labels.json
python generate_test.py

# Download 400 general Banking77 queries and convert to WAV
python download_banking77.py --out ./audio_banking77 --split test --max 400

# Merge the synthetic dataset and tool labels into ./dataset/labels.json
python combine_datasets.py
```

### 3. Run the benchmark

```bash
python benchmark_runner.py \
    --db ./customer_db \
    --dataset ./dataset \
    --models cascade,qwen3omni,nemotron \
    --judge bertscore \
    --max-samples 500 \
    --device cuda
```

---

## HPC Deployment (Singularity/Slurm)

Running vLLM in HPC environments requires specific cache redirection and path bindings to avoid read-only container crashes and CUDA fork corruption.

```bash
singularity exec --nv \
  --env HF_TOKEN="your_token_here" \
  --env VLLM_USE_V1=1 \
  --env HF_HOME=/workspace/tmp_cache/hf \
  --env PIP_CACHE_DIR=/workspace/tmp_cache/pip \
  --env XDG_CACHE_HOME=/workspace/tmp_cache \
  --bind /scratch/users/yourinstitute/yourname/projectdirectory:/workspace \
  --bind /path/to/dataset:/dataset \
  /path/to/vllm-omni.sif \
  /workspace/container_venv/bin/python /workspace/benchmark_runner.py \
    --db /workspace/customer_db \
    --dataset /dataset \
    --models cascade,qwen3omni,nemotron \
    --judge bertscore \
    --max-samples 500
```

---

## Output Structure

```
./results/
├── comparison_report.json       # Cross-model F1 and latency summary
├── qwen3omni/
│   ├── traces.csv               # Per-sample latency + accuracy breakdowns
│   ├── accuracy.json
│   ├── latency.json
│   └── *_synthesis.txt / *_direct.txt
├── nemotron/
│   └── ...
└── cascade/
    └── ...
```

---

## Key Technical Decisions

- **Dynamic Architecture Injection:** `benchmark_runner.py` uses a re-entrant `builtins.__import__` monkey-patch to intercept and register custom multimodal configurations (e.g., `_get_num_multimodal_tokens`) into the `transformers` and `vllm` module registries at runtime, enabling support for experimental Qwen3-Omni architectures not yet merged into upstream libraries.

- **HPC Cache Redirection:** Triton JIT, FlashInfer, and HuggingFace caches are forcibly redirected to `/workspace/tmp_cache/` to prevent `PermissionError` crashes inside strictly sandboxed Singularity containers.

- **Stratified Interleaved Sampling:** The evaluation dataset uses a fixed 20/80 stratification ratio — 100 tool-use lookup queries (5 customer profiles × 20 conversational templates, synthesized with pyttsx3) interleaved with 400 general Banking77 queries (synthesized with gTTS). Files are deterministically shuffled with a fixed seed to guarantee a reproducible mix under `--max-samples` truncation.

- **IVFFlat over Flat:** At 1M records, exact `IndexFlatL2` takes 500ms–5s per search. IVFFlat reduces this to 5–60ms while maintaining >95% recall vs. exact search.

---

## Citation

```bibtex
@article{lee2026whennative,
  title     = {When Do Native Speech-Language Models Beat Cascades?
               A Latency--Accuracy Study of Constrained Telecom Retrieval},
  author    = {Lee, Esther Yi Shan and Ustiugov, Dmitrii and Chen, Wenyan and Liu, Hongrui},
  journal   = {Proceedings of URECA@NTU 2025--26},
  year      = {2026}
}
```

---

## Acknowledgements

This research was conducted under the NTU URECA Undergraduate Research Programme.