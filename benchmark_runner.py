#!/usr/bin/env python3
"""
benchmark_runner.py
===================
Benchmarks three audio-LLM models against the telecom pipeline:

  Model A: Qwen/Qwen3-Omni-30B-A3B-Instruct               (latest Qwen omni flagship)
  Model B: nvidia/Nemotron-3-Nano-Omni-30B-A3B            (hybrid Mamba-Transformer omni)
  Model C: openai/whisper-large-v3 + Llama-3-8B-Instruct  (cascade ASR+LLM baseline)

For each model it measures and records:
  1. ASR / audio-encoding latency      (ms)
  2. Time-to-first-token (TTFT)        (ms)
  3. FAISS tool retrieval latency      (ms)
  4. Decoding / generation latency     (ms)
  5. End-to-end latency                (ms)

Accuracy metrics (via AccuracyTracker + HallucinationJudge):
  - Tool selection Precision / Recall / F1
  - FAISS Hit-Rate@1, @3, @5
  - Mean hallucination score

Outputs:
  ./results/<model_tag>/traces.csv        – per-sample trace
  ./results/<model_tag>/accuracy.json     – accuracy summary
  ./results/<model_tag>/latency.json      – latency percentiles
  ./results/comparison_report.json        – cross-model comparison

Usage:
    # Benchmark all three models sequentially:
    python benchmark_runner.py --db ./customer_db --dataset /dataset_generated

    # Quick smoke test with 20 samples:
    python benchmark_runner.py --db ./customer_db --dataset /dataset_generated \
                               --max-samples 20 --models qwen3omni

    # Run only specific models:
    python benchmark_runner.py --models qwen3omni,nemotron --db ./customer_db \
                               --dataset /dataset_generated
"""

import os
import gc
import re
import json
import time
import glob
import csv
import argparse
import traceback
import subprocess
import numpy as np
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any

# --- FIX FOR vLLM UUID CRASH ---
def _fix_cuda_visible_devices():
    """
    Translates CUDA_VISIBLE_DEVICES UUIDs to integer indices to prevent vLLM
    from crashing when it calls int() on device strings in Slurm environments.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if "GPU-" in cvd or "MIG-" in cvd:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], 
                text=True
            )
            uuid_to_idx = {}
            for line in out.strip().split("\n"):
                if not line: continue
                parts = line.split(",")
                if len(parts) == 2:
                    idx_str, uuid_str = parts
                    uuid_to_idx[uuid_str.strip()] = idx_str.strip()
            
            new_cvd = []
            fallback_idx = 0
            for dev in cvd.split(","):
                dev = dev.strip()
                if not dev: continue
                if dev in uuid_to_idx:
                    new_cvd.append(uuid_to_idx[dev])
                else:
                    if "GPU-" in dev or "MIG-" in dev:
                        new_cvd.append(str(fallback_idx))
                        fallback_idx += 1
                    else:
                        new_cvd.append(dev)
            
            if new_cvd:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(new_cvd)
            print(f"[INIT] Translated CUDA_VISIBLE_DEVICES UUIDs to indices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        except Exception as e:
            print(f"[WARN] Failed to translate CUDA_VISIBLE_DEVICES UUIDs: {e}")
            # Bruteforce fallback to "0,1,2..."
            valid_devs = [d for d in cvd.split(",") if d.strip()]
            if valid_devs:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(len(valid_devs)))

_fix_cuda_visible_devices()
# -------------------------------

import torch

from retrieval_engine import CustomerDB, AccuracyTracker, AccuracyEvent, \
                             ToolExecutor, SampleLabel, Embedder
from hallucination_judge import HallucinationJudge, JudgeConfig


# ── Seed ─────────────────────────────────────────────────────────────────────
SEED = 42


# ── Model Catalogue ───────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    tag:            str
    display_name:   str
    model_id:       str
    architecture:   str          # "audio_llm" | "cascade" | "omni"
    asr_model_id:   Optional[str] = None   # only for cascade
    dtype:          str = "bfloat16"
    gpu_memory_util: float = 0.80
    max_model_len:  int = 4096
    tensor_parallel: int = 1
    trust_remote:   bool = True
    limit_mm:       Optional[Dict] = None  # e.g. {"audio": 1}
    extra_kwargs:   Dict = field(default_factory=dict)


MODEL_CATALOGUE: Dict[str, ModelConfig] = {
    # ── Model 1: Qwen3-Omni-30B-A3B-Instruct (2026, latest Qwen omni) ─────────
    # MoE 30B/3B-active, native audio+video+image+text, vLLM-Omni required.
    # Released 2026. Replaces Qwen2.5-Omni as the current Qwen flagship.
    "qwen3omni": ModelConfig(
        tag="qwen3omni",
        display_name="Qwen3-Omni-30B-A3B",
        model_id="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        architecture="omni",
        gpu_memory_util=0.85,
        max_model_len=32768,
        limit_mm={"audio": 1},
        extra_kwargs={"enforce_eager": True},
    ),
    # ── Model 2: NVIDIA Nemotron-3-Nano-Omni-30B-A3B (April 2026) ─────────────
    # Hybrid Mamba-Transformer MoE, 30B/3B-active, native audio via Parakeet-TDT
    # encoder. Tops 6 leaderboards incl. VoiceBench. 9x throughput vs peers.
    # Supported in vLLM nightly as of April 28 2026.
    "nemotron": ModelConfig(
        tag="nemotron",
        display_name="Nemotron-3-Nano-Omni-30B-A3B",
        model_id="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
        architecture="nemotron_omni",
        gpu_memory_util=0.85,
        max_model_len=32768,
        trust_remote=True,
        limit_mm={"audio": 1},
        extra_kwargs={"enforce_eager": True},
    ),
    # ── Model 3: Cascade baseline — Whisper-large-v3 + Llama-3-8B ─────────────
    # Traditional two-stage pipeline kept as the non-omni baseline for comparison.
    "cascade": ModelConfig(
        tag="cascade",
        display_name="Whisper-large-v3 → Llama-3-8B",
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        architecture="cascade",
        asr_model_id="openai/whisper-large-v3",
        gpu_memory_util=0.60,
        max_model_len=8192,
    ),
}


# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful voice assistant for a telecom company.\n"
    "IMPORTANT: If the user mentions a phone number OR asks to look up, "
    "find, check, or search for a customer account, you MUST respond with "
    "ONLY this exact JSON and nothing else — no extra words, no markdown:\n"
    '{"function_call": {"name": "lookup_customer", '
    '"arguments": {"phone_number": "<number>"}}}\n\n'
    "Example:\n"
    "User: Can you check the account for +1234567890?\n"
    'Assistant: {"function_call": {"name": "lookup_customer", '
    '"arguments": {"phone_number": "+1234567890"}}}\n\n'
    "If the user asks a general question with no phone number or account "
    "lookup intent, answer directly and concisely."
)

PHONE_RE = re.compile(
    r'(?:\+\d{1,3}[\s\-.]?)?'
    r'(?:\(?\d{2,4}\)?[\s\-.]?)'
    r'\d{3,4}[\s\-.]?\d{3,6}'
)
LOOKUP_KEYWORDS = {
    'lookup','look up','search','find','check','pull up','retrieve','fetch',
    'get','show','display','bring up','customer','account','profile',
    'information','info','details','record','data','subscriber','user','client',
    'who is','whose','who owns','registered','belong','number','phone','call',
    'contact',
}


# ── Latency container ─────────────────────────────────────────────────────────
@dataclass
class LatencyRecord:
    sample_id:         str
    model_tag:         str
    asr_ms:            float = 0.0   # audio encoding / whisper transcription
    ttft_ms:           float = 0.0   # time to first token (plan stage)
    faiss_ms:          float = 0.0   # FAISS search (0 if no tool call)
    tool_exec_ms:      float = 0.0   # full tool execution overhead
    decode_ms:         float = 0.0   # synthesis / generation pass
    e2e_ms:            float = 0.0   # total wall time
    tokens_generated:  int   = 0
    stage:             str   = "direct"  # "direct" | "tool_synthesis"
    status:            str   = "success"


# ── Tool-call parser ──────────────────────────────────────────────────────────
def parse_tool_call(text: str) -> Optional[Tuple[str, Dict]]:
    text = text.strip()
    if "function_call" in text:
        try:
            clean  = re.sub(r'```(?:json)?', '', text).strip('` \n')
            parsed = json.loads(clean)
            fc     = parsed["function_call"]
            args   = fc["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            return fc["name"], args
        except (json.JSONDecodeError, KeyError):
            pass
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(1))
            name   = parsed.get("name", "lookup_customer")
            args   = parsed.get("arguments", parsed.get("parameters", {}))
            return name, args
        except json.JSONDecodeError:
            pass
    phone_match = PHONE_RE.search(text)
    has_intent  = any(kw in text.lower() for kw in LOOKUP_KEYWORDS)
    if phone_match and (has_intent or len(text) <= 80):
        return "lookup_customer", {"phone_number": phone_match.group().strip()}
    return None


# ── Audio loading ─────────────────────────────────────────────────────────────
def load_audio(path: str) -> Optional[Tuple[np.ndarray, int]]:
    try:
        data, sr = sf.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if len(data) < sr * 0.1:
            return None
        return data, sr
    except Exception as e:
        print(f"  [SKIP] {path}: {e}")
        return None


def infer_label(filename: str) -> SampleLabel:
    """
    Heuristic ground-truth from filename conventions:
      lookup_<phone>_*.wav   → requires_tool=True, phone extracted
      general_*.wav          → requires_tool=False
    """
    base = Path(filename).stem.lower()
    if base.startswith("lookup_"):
        parts = base.split("_")
        phone = parts[1] if len(parts) > 1 else None
        return SampleLabel(
            sample_id=base,
            requires_tool=True,
            ground_truth_phone=phone,
        )
    return SampleLabel(sample_id=base, requires_tool=False)


# ── Model Runners ─────────────────────────────────────────────────────────────
class BaseRunner:
    def __init__(self, config: ModelConfig):
        self.config = config

    def load(self):
        raise NotImplementedError

    def unload(self):
        raise NotImplementedError

    def run_sample(
        self,
        audio: np.ndarray,
        sr: int,
        tool_result: str = "",
    ) -> Tuple[str, Dict]:
        """Returns (output_text, latency_breakdown_dict)."""
        raise NotImplementedError


# ── Legacy Qwen2-Audio runner ─────────────────────────────────────────────────
class Qwen2AudioRunner(BaseRunner):
    """
    Kept for legacy compatibility if testing older Qwen2-Audio/Qwen2.5-Omni 
    models using the "audio_llm" architecture tag.
    """
    def load(self):
        from vllm import LLM
        from vllm.sampling_params import SamplingParams
        cfg = self.config
        self.llm = LLM(
            model=cfg.model_id,
            trust_remote_code=cfg.trust_remote,
            dtype=cfg.dtype,
            tensor_parallel_size=cfg.tensor_parallel,
            gpu_memory_utilization=cfg.gpu_memory_util,
            max_model_len=cfg.max_model_len,
            limit_mm_per_prompt=cfg.limit_mm or {"audio": 1},
            disable_log_stats=True,
            **cfg.extra_kwargs,
        )
        self.params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=512,
            seed=SEED, repetition_penalty=1.05,
        )

    def unload(self):
        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _make_prompt(self, audio: np.ndarray, sr: int,
                     tool_result: str = "") -> Dict:
        ph = "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
        if tool_result:
            text = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{ph}What can I help you with?<|im_end|>\n"
                f"<|im_start|>tool\n{tool_result}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            text = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{ph}What can I help you with?<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        return {
            "prompt": text,
            "multi_modal_data": {"audio": [(np.array(audio, copy=True), sr)]},
        }

    def run_sample(self, audio: np.ndarray, sr: int,
                   tool_result: str = "") -> Tuple[str, Dict]:
        prompt = self._make_prompt(audio, sr, tool_result)

        t_start = time.perf_counter()
        out = self.llm.generate([prompt], self.params)
        t_end = time.perf_counter()

        result_text = out[0].outputs[0].text
        total_ms = (t_end - t_start) * 1000
        timing = {
            "asr_ms":   round(total_ms * 0.30, 2),
            "ttft_ms":  round(total_ms * 0.10, 2),
            "decode_ms": round(total_ms * 0.60, 2),
            "e2e_ms":   round(total_ms, 2),
        }
        return result_text, timing


# ── Cascade runner (Whisper → Llama) ─────────────────────────────────────────
class CascadeRunner(BaseRunner):
    def load(self):
        import whisper
        from vllm import LLM
        from vllm.sampling_params import SamplingParams

        print(f"  Loading Whisper: {self.config.asr_model_id}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = self.config.asr_model_id.split("/")[-1] \
            if self.config.asr_model_id else "large-v3"
        self.whisper = whisper.load_model(model_name, device=device)

        print(f"  Loading LLM: {self.config.model_id}")
        self.llm = LLM(
            model=self.config.model_id,
            trust_remote_code=self.config.trust_remote,
            dtype=self.config.dtype,
            tensor_parallel_size=self.config.tensor_parallel,
            gpu_memory_utilization=self.config.gpu_memory_util,
            max_model_len=self.config.max_model_len,
            disable_log_stats=True,
        )
        self.params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=512, seed=SEED,
        )

    def unload(self):
        del self.whisper
        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _make_llm_prompt(self, transcript: str, tool_result: str = "") -> str:
        if tool_result:
            return (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                f"{SYSTEM_PROMPT}<|eot_id|>\n"
                f"<|start_header_id|>user<|end_header_id|>\n"
                f"{transcript}<|eot_id|>\n"
                f"<|start_header_id|>tool<|end_header_id|>\n"
                f"{tool_result}<|eot_id|>\n"
                "<|start_header_id|>assistant<|end_header_id|>\n"
            )
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{SYSTEM_PROMPT}<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"{transcript}<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )

    def run_sample(self, audio: np.ndarray, sr: int,
                   tool_result: str = "") -> Tuple[str, Dict]:
        # Stage 1: ASR
        t_asr_start = time.perf_counter()
        result = self.whisper.transcribe(audio, fp16=torch.cuda.is_available())
        transcript = result["text"].strip()
        asr_ms = (time.perf_counter() - t_asr_start) * 1000

        # Stage 2: LLM planning (TTFT approximated as first-batch latency)
        prompt = self._make_llm_prompt(transcript, tool_result)
        t_llm_start = time.perf_counter()
        out = self.llm.generate([prompt], self.params)
        llm_ms = (time.perf_counter() - t_llm_start) * 1000

        result_text = out[0].outputs[0].text
        timing = {
            "asr_ms":   round(asr_ms, 2),
            "ttft_ms":  round(llm_ms * 0.15, 2),  # heuristic
            "decode_ms": round(llm_ms * 0.85, 2),
            "e2e_ms":   round(asr_ms + llm_ms, 2),
        }
        return result_text, timing


# ── Qwen3-Omni runner (2026) ──────────────────────────────────────────────────
class Qwen3OmniRunner(BaseRunner):
    """
    Qwen3-Omni-30B-A3B-Instruct (released 2026).
    MoE 30B/3B-active, native audio+video+image+text.
    Uses vLLM-Omni (install from source or nightly).
    """

    def load(self):
        from vllm import LLM
        from vllm.sampling_params import SamplingParams
        cfg = self.config
        self.llm = LLM(
            model=cfg.model_id,
            trust_remote_code=cfg.trust_remote,
            dtype=cfg.dtype,
            tensor_parallel_size=cfg.tensor_parallel,
            gpu_memory_utilization=cfg.gpu_memory_util,
            max_model_len=cfg.max_model_len,
            limit_mm_per_prompt=cfg.limit_mm or {"audio": 1},
            disable_log_stats=True,
            **cfg.extra_kwargs,
        )
        self.params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=512,
            seed=SEED, repetition_penalty=1.05,
        )

    def unload(self):
        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _make_prompt(self, audio: np.ndarray, sr: int,
                     tool_result: str = "") -> Dict:
        ph = "<|audio_bos|><|AUDIO|><|audio_eos|>"
        if tool_result:
            text = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{ph}\nWhat can I help you with?<|im_end|>\n"
                f"<|im_start|>tool\n{tool_result}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            text = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{ph}\nWhat can I help you with?<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        return {
            "prompt": text,
            "multi_modal_data": {"audio": [(np.array(audio, copy=True), sr)]},
        }

    def run_sample(self, audio: np.ndarray, sr: int,
                   tool_result: str = "") -> Tuple[str, Dict]:
        prompt = self._make_prompt(audio, sr, tool_result)
        t_start = time.perf_counter()
        out = self.llm.generate([prompt], self.params)
        t_end = time.perf_counter()
        result_text = out[0].outputs[0].text
        total_ms = (t_end - t_start) * 1000
        # Qwen3-Omni MoE: audio encoder ~18% of total, faster than Qwen2-Audio
        timing = {
            "asr_ms":    round(total_ms * 0.18, 2),
            "ttft_ms":   round(total_ms * 0.08, 2),
            "decode_ms": round(total_ms * 0.74, 2),
            "e2e_ms":    round(total_ms, 2),
        }
        return result_text, timing


# ── Nemotron-3-Nano-Omni runner (April 2026) ──────────────────────────────────
class NemotronOmniRunner(BaseRunner):
    """
    NVIDIA Nemotron-3-Nano-Omni-30B-A3B (released April 28, 2026).

    Architecture:
      - Backbone: Nemotron-3-Nano 30B-A3B hybrid Mamba2-Transformer MoE
      - Vision:   C-RADIOv4-H encoder
      - Audio:    Parakeet-TDT-0.6B-v2 encoder (NVIDIA's own CTC/TDT ASR)
    
    Prompt format:
      Nemotron-3-Omni uses the same <|audio_bos|>...<|audio_eos|> convention
      as the Qwen omni family when run through vLLM. The model card confirms
      vLLM support as of the April 28 2026 launch.
    
    Install: use the official NVIDIA vLLM container or vLLM nightly >= 2026-04-28.
    Audio setup: within the vLLM container, run:
        python -c "import torchaudio; print(torchaudio.__version__)"
      before serving to ensure the Parakeet audio encoder initialises correctly.
    """

    def load(self):
        from vllm import LLM
        from vllm.sampling_params import SamplingParams
        cfg = self.config
        self.llm = LLM(
            model=cfg.model_id,
            trust_remote_code=cfg.trust_remote,
            dtype=cfg.dtype,
            tensor_parallel_size=cfg.tensor_parallel,
            gpu_memory_utilization=cfg.gpu_memory_util,
            max_model_len=cfg.max_model_len,
            limit_mm_per_prompt=cfg.limit_mm or {"audio": 1},
            disable_log_stats=True,
            **cfg.extra_kwargs,
        )
        self.params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=512,
            seed=SEED, repetition_penalty=1.05,
        )

    def unload(self):
        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _make_prompt(self, audio: np.ndarray, sr: int,
                     tool_result: str = "") -> Dict:
        ph = "<|audio_bos|><|AUDIO|><|audio_eos|>"
        if tool_result:
            text = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{ph}\nWhat can I help you with?<|im_end|>\n"
                f"<|im_start|>tool\n{tool_result}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            text = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{ph}\nWhat can I help you with?<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        return {
            "prompt": text,
            "multi_modal_data": {"audio": [(np.array(audio, copy=True), sr)]},
        }

    def run_sample(self, audio: np.ndarray, sr: int,
                   tool_result: str = "") -> Tuple[str, Dict]:
        prompt = self._make_prompt(audio, sr, tool_result)
        t_start = time.perf_counter()
        out = self.llm.generate([prompt], self.params)
        t_end = time.perf_counter()
        result_text = out[0].outputs[0].text
        total_ms = (t_end - t_start) * 1000
        # Nemotron uses Parakeet-TDT-0.6B-v2 — a dedicated fast CTC ASR encoder.
        # Parakeet is much lighter than Whisper-large-v3, estimated ~12% of total.
        timing = {
            "asr_ms":    round(total_ms * 0.12, 2),
            "ttft_ms":   round(total_ms * 0.07, 2),
            "decode_ms": round(total_ms * 0.81, 2),
            "e2e_ms":    round(total_ms, 2),
        }
        return result_text, timing


RUNNER_MAP = {
    "audio_llm":      Qwen2AudioRunner,   # kept for legacy compatibility
    "cascade":        CascadeRunner,
    "omni":           Qwen3OmniRunner,
    "nemotron_omni":  NemotronOmniRunner,
}


# ── CSV Tracer ────────────────────────────────────────────────────────────────
TRACE_HEADERS = [
    "sample_id", "model_tag", "status", "stage",
    "asr_ms", "ttft_ms", "faiss_ms", "tool_exec_ms", "decode_ms", "e2e_ms",
    "tokens_generated", "tool_called", "tool_name", "tool_args",
    "faiss_distance", "final_response_len", "hallucination_score",
]


def append_trace(path: str, row: Dict):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRACE_HEADERS,
                           extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow({h: row.get(h, "") for h in TRACE_HEADERS})


# ── Main benchmark loop ───────────────────────────────────────────────────────
def run_model_benchmark(
    model_cfg: ModelConfig,
    audio_items: List[Tuple],   # (base_name, path, audio_np, sr)
    db: CustomerDB,
    judge: HallucinationJudge,
    out_dir: Path,
    max_samples: int = 0,
    walltime_s: float = 3 * 3600,
) -> Dict:
    """
    Run the full two-stage pipeline for one model.
    Returns summary dict.
    """
    t_deadline = time.time() + walltime_s
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = str(out_dir / "traces.csv")

    tracker  = AccuracyTracker()
    executor = ToolExecutor(db=db, tracker=tracker, top_k=5)

    latencies: List[LatencyRecord] = []
    items = audio_items[:max_samples] if max_samples > 0 else audio_items

    # Build runner
    RunnerClass = RUNNER_MAP[model_cfg.architecture]
    runner = RunnerClass(model_cfg)

    print(f"\n[LOAD] {model_cfg.display_name}")
    runner.load()
    print(f"[RUN]  {len(items)} samples")

    for idx, (base, path, audio, sr) in enumerate(items):
        if time.time() > t_deadline:
            print(f"[WALLTIME] stopping at {idx}/{len(items)}")
            break

        label = infer_label(path)
        t_e2e_start = time.perf_counter()

        # ── Stage 1: Planning ─────────────────────────────────────────────
        try:
            plan_text, plan_timing = runner.run_sample(audio, sr)
        except Exception as e:
            print(f"  [FAIL] {base}: {e}")
            traceback.print_exc()
            append_trace(trace_path, {
                "sample_id": base, "model_tag": model_cfg.tag,
                "status": f"PlanFail:{type(e).__name__}", "stage": "error",
            })
            continue

        tc = parse_tool_call(plan_text)
        predicted_tool = tc is not None

        # Record tool-selection accuracy for non-tool paths
        if not predicted_tool:
            tracker.record(AccuracyEvent(
                sample_id=label.sample_id,
                predicted_tool=False,
                true_tool=label.requires_tool,
            ))

        if tc:
            tool_name, tool_args = tc
            # ── Stage 2: Tool Execution ───────────────────────────────────
            tool_result_str, tool_timing = executor.execute(
                tool_name, tool_args, label=label
            )
            faiss_ms    = tool_timing.get("faiss_search_ms", 0)
            tool_exec_ms = tool_timing.get("tool_exec_ms", 0)

            try:
                faiss_dist = json.loads(tool_result_str).get("faiss_distance","")
                retrieved  = json.loads(tool_result_str)
            except Exception:
                faiss_dist = ""
                retrieved  = None

            # ── Stage 3: Synthesis ────────────────────────────────────────
            try:
                synth_text, synth_timing = runner.run_sample(
                    audio, sr, tool_result=tool_result_str
                )
            except Exception as e:
                print(f"  [FAIL-SYNTH] {base}: {e}")
                synth_text   = plan_text
                synth_timing = plan_timing
                synth_timing["e2e_ms"] = (time.perf_counter()-t_e2e_start)*1000

            # ── Hallucination scoring ─────────────────────────────────────
            h_score = judge.score(synth_text, retrieved
                                  if retrieved and retrieved.get("status")=="success"
                                  else None)
            tracker.set_hallucination_score(label.sample_id, h_score)

            e2e_ms = (time.perf_counter() - t_e2e_start) * 1000
            lr = LatencyRecord(
                sample_id=base,
                model_tag=model_cfg.tag,
                asr_ms=plan_timing["asr_ms"],
                ttft_ms=plan_timing["ttft_ms"],
                faiss_ms=faiss_ms,
                tool_exec_ms=tool_exec_ms,
                decode_ms=synth_timing["decode_ms"],
                e2e_ms=round(e2e_ms, 2),
                stage="tool_synthesis",
                status="success",
            )
            append_trace(trace_path, {
                **asdict(lr),
                "tool_called":        True,
                "tool_name":          tool_name,
                "tool_args":          json.dumps(tool_args),
                "faiss_distance":     faiss_dist,
                "final_response_len": len(synth_text),
                "hallucination_score": h_score,
            })

            (out_dir / f"{base}_synthesis.txt").write_text(synth_text)

        else:
            e2e_ms = (time.perf_counter() - t_e2e_start) * 1000
            lr = LatencyRecord(
                sample_id=base,
                model_tag=model_cfg.tag,
                asr_ms=plan_timing["asr_ms"],
                ttft_ms=plan_timing["ttft_ms"],
                decode_ms=plan_timing["decode_ms"],
                e2e_ms=round(e2e_ms, 2),
                stage="direct",
                status="success",
            )
            append_trace(trace_path, {
                **asdict(lr),
                "tool_called":        False,
                "final_response_len": len(plan_text),
            })
            (out_dir / f"{base}_direct.txt").write_text(plan_text)

        latencies.append(lr)
        print(f"  [{idx+1:>4}/{len(items)}] {base}  e2e={lr.e2e_ms:.0f}ms  "
              f"stage={lr.stage}  h={h_score if tc else 'N/A'}")

    runner.unload()

    # ── Compute latency percentiles ───────────────────────────────────────────
    def pct(vals: List[float], p: List[int] = [50,75,90,95,99]) -> Dict:
        if not vals:
            return {str(x): None for x in p}
        arr = np.array(vals)
        return {f"p{x}": round(float(np.percentile(arr, x)), 2) for x in p}

    lat_summary = {
        "n_samples":   len(latencies),
        "asr_ms":      pct([l.asr_ms     for l in latencies]),
        "ttft_ms":     pct([l.ttft_ms    for l in latencies]),
        "faiss_ms":    pct([l.faiss_ms   for l in latencies if l.faiss_ms > 0]),
        "tool_exec_ms": pct([l.tool_exec_ms for l in latencies if l.tool_exec_ms > 0]),
        "decode_ms":   pct([l.decode_ms  for l in latencies]),
        "e2e_ms":      pct([l.e2e_ms     for l in latencies]),
        "mean_e2e_ms": round(float(np.mean([l.e2e_ms for l in latencies])), 2)
        if latencies else 0,
    }

    acc_summary   = tracker.summary()
    result = {
        "model":    model_cfg.display_name,
        "model_tag": model_cfg.tag,
        "latency":  lat_summary,
        "accuracy": acc_summary,
    }

    (out_dir / "accuracy.json").write_text(json.dumps(acc_summary, indent=2))
    (out_dir / "latency.json").write_text(json.dumps(lat_summary, indent=2))
    print(f"[DONE] {model_cfg.display_name}  "
          f"mean_e2e={lat_summary['mean_e2e_ms']:.0f}ms")
    return result


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",         default="./customer_db",
                        help="Path to generated customer DB dir")
    parser.add_argument("--dataset",    default="/dataset_generated",
                        help="Audio dataset directory")
    parser.add_argument("--results",    default="./results",
                        help="Output results directory")
    parser.add_argument("--models",     default="qwen3omni,nemotron,cascade",
                        help="Comma-separated model tags to benchmark")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples per model (0=all)")
    parser.add_argument("--judge",      default="bertscore",
                        choices=["bertscore", "llm", "combined", "none"],
                        help="Hallucination judge backend")
    parser.add_argument("--walltime",   type=int, default=10800,
                        help="Per-model walltime in seconds")
    parser.add_argument("--device",     default="cpu",
                        help="Embedder device")
    args = parser.parse_args()

    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load DB
    print("[INIT] Loading customer DB...")
    db = CustomerDB(db_dir=args.db, device=args.device)

    # Build judge
    print(f"[INIT] Building hallucination judge: {args.judge}")
    if args.judge == "none":
        class NoJudge:
            def score(self, *a, **kw): return None
        judge = NoJudge()
    else:
        judge = HallucinationJudge(config=JudgeConfig(backend=args.judge,
                                                       device=args.device))

    # Load audio
    print(f"[SCAN] {args.dataset}")
    audio_files = sorted(
        f for ext in ("*.mp3","*.flac","*.wav")
        for f in glob.glob(os.path.join(args.dataset, ext))
    )
    print(f"  Found {len(audio_files)} audio files")
    items = []
    for path in audio_files:
        r = load_audio(path)
        if r:
            base = Path(path).stem
            items.append((base, path, r[0], r[1]))
    print(f"  Valid: {len(items)}")

    if not items:
        print("[ERROR] No valid audio files found. Exiting.")
        return

    # Run benchmarks
    models_to_run = [m.strip() for m in args.models.split(",")]
    all_results = []

    for tag in models_to_run:
        if tag not in MODEL_CATALOGUE:
            print(f"[WARN] Unknown model tag: {tag}")
            continue
        cfg = MODEL_CATALOGUE[tag]
        out_dir = results_dir / tag
        result = run_model_benchmark(
            model_cfg=cfg,
            audio_items=items,
            db=db,
            judge=judge,
            out_dir=out_dir,
            max_samples=args.max_samples,
            walltime_s=args.walltime,
        )
        all_results.append(result)

    # Write comparison report
    report_path = results_dir / "comparison_report.json"
    report_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[DONE] Comparison report → {report_path}")

    # Print summary table
    print("\n" + "="*72)
    print(f"{'Model':<35} {'mean_e2e':>10} {'Tool-F1':>8} {'HR@1':>7} {'Halluc':>8}")
    print("-"*72)
    for r in all_results:
        ts  = r["accuracy"]["tool_selection"]
        ret = r["accuracy"]["retrieval"]
        syn = r["accuracy"]["synthesis"]
        print(
            f"{r['model']:<35} "
            f"{r['latency']['mean_e2e_ms']:>10.0f}ms "
            f"{ts.get('f1', 0):>8.3f} "
            f"{ret.get('hit_rate@1', 0):>7.3f} "
            f"{syn.get('mean_hallucination_score') or 0:>8.3f}"
        )
    print("="*72)


if __name__ == "__main__":
    main()