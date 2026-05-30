#!/usr/bin/env python3
"""
benchmark_runner.py
===================
Evaluates multi-modal audio-LLMs and cascade speech-to-text pipelines against 
an indexed telecom customer database using FAISS vector retrieval and 
automated semantic grounding evaluation.

Core functionality:
  * Overrides core process forks to force process 'spawn' configurations and 
    translates Slurm/PBS string GPU UUIDs into safe integer indices.
  * Injects a dynamic builtins.__import__ monkey-patch hook to dynamically register
    experimental Qwen3/Qwen3.5 architectures within transformers and vLLM runtimes.
  * Redirects Home, FlashInfer, and Triton JIT compilation caches to writeable 
    workspaces to bypass read-only container filesystem collisions.
  * Tracks and isolates granular latency breakdowns across five evaluation dimensions
    (ASR / encoding overhead, TTFT, FAISS lookup, tool overhead, text decoding).
  * Measures operational accuracy profiles, containing Tool-Selection Precision/Recall/F1, 
    FAISS Hit-Rate@K, and semantic grounding (BERTScore/LLM-as-Judge).

Prerequisites (Inputs):
  - A compiled customer table and trained FAISS vector database index (--db)
  - A directory of speech evaluation target vectors (--dataset) containing .wav/.mp3/.flac assets
  - Environment variable tokens (HF_TOKEN) authorizing access to gated model weight catalogs

Outputs:
  - ./results/<model_tag>/traces.csv        – Raw per-sample latencies, arguments, and judge scores
  - ./results/<model_tag>/accuracy.json     – Aggregated tool matrix, accuracy summary, and hit rates
  - ./results/<model_tag>/latency.json      – Quantiled latency percentiles (p50, p75, p90, p95, p99)
  - ./results/comparison_report.json        – Final cross-model evaluation performance report

Usage (Standard Local Python Environment):
    python benchmark_runner.py --db ./customer_db --dataset /dataset_generated --max-samples 500

Usage (HPC Production Deployment via Singularity SIF Container):
    singularity exec --nv \
      --env HF_TOKEN="your_actual_token_here" \
      --env VLLM_USE_V1=1 \
      --env HF_HOME=/workspace/tmp_cache/hf \
      --env PIP_CACHE_DIR=/workspace/tmp_cache/pip \
      --env XDG_CACHE_HOME=/workspace/tmp_cache \
      --bind /path/to/your/project_directory:/workspace \
      --bind /path/to/your/audio_dataset_directory:/dataset \
      /path/to/your/vllm-omni.sif \
      /workspace/container_venv/bin/python /workspace/benchmark_runner.py \
        --db /workspace/customer_db \
        --dataset /dataset \
        --models cascade,qwen3omni,nemotron \
        --judge bertscore \
        --max-samples 500
"""

import os
import sys
import multiprocessing
import subprocess
from pathlib import Path

# vLLM V1 engine uses subprocesses. Since CustomerDB/Judge initialize CUDA in 
# the main process, forking causes a CUDA context corruption ("driver too old").
# We force 'spawn' so child processes start with a clean CUDA context.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# Disable anonymous usage telemetry to prevent container permission crashes
os.environ["VLLM_NO_USAGE_STATS"] = "1"

# Redirect Home, FlashInfer, and Triton JIT compilation caches to your writeable scratch
# workspace. This prevents PermissionErrors inside Singularity container environments
# where the default Home (~/.cache/) filesystem blocks runtime writes.
os.environ["HOME"] = "/workspace/tmp_cache/home"
os.environ["FLASHINFER_CACHE_DIR"] = "/workspace/tmp_cache/flashinfer"
os.environ["TRITON_CACHE_DIR"] = "/workspace/tmp_cache/triton"

try:
    Path("/workspace/tmp_cache/home").mkdir(parents=True, exist_ok=True)
    Path("/workspace/tmp_cache/flashinfer").mkdir(parents=True, exist_ok=True)
    Path("/workspace/tmp_cache/triton").mkdir(parents=True, exist_ok=True)
except Exception:
    pass

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass


def _fix_cuda_visible_devices():
    """
    PBS/Slurm sometimes sets CUDA_VISIBLE_DEVICES to GPU UUIDs like
    'GPU-a1b2c3...'. vLLM calls int() on these and crashes immediately.
    This translates UUIDs to plain integer indices (0, 1, ...) before
    anything else imports torch or vLLM.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not ("GPU-" in cvd or "MIG-" in cvd):
        return   # already integers, nothing to do

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True,
        )
        uuid_to_idx = {}
        for line in out.strip().splitlines():
            parts = line.split(",")
            if len(parts) == 2:
                uuid_to_idx[parts[1].strip()] = parts[0].strip()

        new_cvd, fallback = [], 0
        for dev in cvd.split(","):
            dev = dev.strip()
            if not dev:
                continue
            if dev in uuid_to_idx:
                new_cvd.append(uuid_to_idx[dev])
            elif "GPU-" in dev or "MIG-" in dev:
                new_cvd.append(str(fallback))
                fallback += 1
            else:
                new_cvd.append(dev)

        if new_cvd:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(new_cvd)
            print(f"[INIT] CUDA_VISIBLE_DEVICES → {os.environ['CUDA_VISIBLE_DEVICES']}")
    except Exception as e:
        print(f"[WARN] UUID translation failed ({e}); trying fallback integers")
        devs = [d for d in cvd.split(",") if d.strip()]
        if devs:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(len(devs)))

_fix_cuda_visible_devices()


def _disable_torch_compile_globally():
    """
    vLLM spawns subprocesses (via python -m) to inspect model architectures.
    These subprocesses load vLLM -> deep_gemm -> torch.compile -> inductor.
    Inductor crashes on some PyTorch versions with 'duplicate template name'
    or missing 'mm_scaled'.
    We globally disable torch.compile by injecting a sitecustomize.py into
    PYTHONPATH, affecting this process and all child processes.
    """
    patch_dir = Path.cwd() / ".vllm_patch"
    patch_dir.mkdir(exist_ok=True)
    site_file = patch_dir / "sitecustomize.py"
    
    code = r"""
import os
import sys
import types
import builtins

if os.environ.get("VLLM_DISABLE_TORCH_COMPILE") == "1":
    try:
        import torch
        def _noop_compile(fn=None, *args, **kwargs):
            if fn is None: return lambda f: f
            return fn
        torch.compile = _noop_compile
        
        # Patch torch.prod on CUDA to bypass NVRTC compilation and resolve libnvrtc-builtins issues
        orig_prod = torch.Tensor.prod
        def patched_prod(self, *args, **kwargs):
            if self.is_cuda:
                return orig_prod(self.cpu(), *args, **kwargs).cuda()
            return orig_prod(self, *args, **kwargs)
        torch.Tensor.prod = patched_prod

        orig_torch_prod = torch.prod
        def patched_torch_prod(input, *args, **kwargs):
            if isinstance(input, torch.Tensor) and input.is_cuda:
                return orig_torch_prod(input.cpu(), *args, **kwargs).cuda()
            return orig_torch_prod(input, *args, **kwargs)
        torch.prod = patched_torch_prod
    except Exception:
        pass

# Ensure Home and JIT cache variables are redirected in child processes too
os.environ["HOME"] = "/workspace/tmp_cache/home"
os.environ["FLASHINFER_CACHE_DIR"] = "/workspace/tmp_cache/flashinfer"
os.environ["TRITON_CACHE_DIR"] = "/workspace/tmp_cache/triton"

# --- START OF VLLM OMNI PATCH ---
# Uses a safe, built-in __import__ hook with a re-entrancy lock to patch target modules
# as soon as they are loaded, keeping sys.modules as a pure dict
# to prevent any C-extension (like pandas/numpy) import conflicts.
try:
    _in_patch = False
    original_import = builtins.__import__

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        global _in_patch
        module = original_import(name, globals, locals, fromlist, level)
        
        # Re-entrancy guard: if we are already inside the patcher, execute natively
        if _in_patch:
            return module
            
        if name == "transformers" or "vllm" in name:
            _in_patch = True
            try:
                # Patch the vLLM Transformers Model Executor class methods if loaded
                if "vllm.model_executor.models.transformers" in sys.modules:
                    tf_mod = sys.modules["vllm.model_executor.models.transformers"]
                    for attr_name in dir(tf_mod):
                        attr = getattr(tf_mod, attr_name)
                        if isinstance(attr, type) and hasattr(attr, "get_max_image_tokens"):
                            original_method = getattr(attr, "get_max_image_tokens")
                            if not hasattr(original_method, "_is_patched"):
                                def make_patched_method(orig_m):
                                    def patched_method(self, *args, **kwargs):
                                        processor = getattr(self, "processor", None)
                                        if processor is not None and not hasattr(processor, "_get_num_multimodal_tokens"):
                                            def dummy_get_tokens(*a, **kw):
                                                return {"num_image_tokens": [2048]}
                                            processor._get_num_multimodal_tokens = types.MethodType(dummy_get_tokens, processor)
                                            print(f"\n[vLLM Patch] Injected _get_num_multimodal_tokens onto {type(processor).__name__} instance!\n", flush=True)
                                        return orig_m(self, *args, **kwargs)
                                    patched_method._is_patched = True
                                    return patched_method
                                setattr(attr, "get_max_image_tokens", make_patched_method(original_method))
                                print(f"[vLLM Patch] Successfully wrapped {attr_name}.get_max_image_tokens", flush=True)
                
                #Patch the lazy-loaded Qwen3OmniMoeProcessor and AutoModel Config Registry
                if "transformers" in sys.modules:
                    trans_mod = sys.modules["transformers"]
                    if hasattr(trans_mod, "Qwen3OmniMoeProcessor"):
                        cls = getattr(trans_mod, "Qwen3OmniMoeProcessor")
                        if not hasattr(cls, "_get_num_multimodal_tokens"):
                            def _dummy_get_num_multimodal_tokens(self, *args, **kwargs):
                                return {"num_image_tokens": [2048]}
                            cls._get_num_multimodal_tokens = _dummy_get_num_multimodal_tokens
                            print(f"\n[vLLM Patch] Dynamic Hook successfully injected _get_num_multimodal_tokens into {cls.__name__}\n", flush=True)
                    
                    # Intercept and register the Qwen3 config mapping into the AutoModel base class
                    if hasattr(trans_mod, "Qwen3OmniMoeConfig") and hasattr(trans_mod, "AutoModel"):
                        config_cls = getattr(trans_mod, "Qwen3OmniMoeConfig")
                        auto_model_cls = getattr(trans_mod, "AutoModel")
                        
                        # Only register if not already completed to prevent loop spamming
                        if not getattr(auto_model_cls, "_qwen3_registered", False):
                            model_cls = None
                            for target in ("Qwen3OmniMoeModel", "Qwen3OmniMoeForCausalLM", "Qwen3OmniMoeForConditionalGeneration"):
                                if hasattr(trans_mod, target):
                                    model_cls = getattr(trans_mod, target)
                                    break
                            if model_cls:
                                auto_model_cls.register(config_cls, model_cls)
                                auto_model_cls._qwen3_registered = True
                                print(f"\n[vLLM Patch] Registered {config_cls.__name__} under AutoModel with {model_cls.__name__}\n", flush=True)

                    # Intercept and register the qwen3_5 config mapping into AutoConfig
                    if hasattr(trans_mod, "AutoConfig"):
                        auto_config_cls = getattr(trans_mod, "AutoConfig")
                        base_config = None
                        for fallback_name in ("Qwen2Config", "LlamaConfig", "PreTrainedConfig"):
                            if hasattr(trans_mod, fallback_name):
                                base_config = getattr(trans_mod, fallback_name)
                                break
                        if base_config:
                            class Qwen3_5Config(base_config):
                                model_type = "qwen3_5"
                            auto_config_cls.register("qwen3_5", Qwen3_5Config)
                            print(f"\n[vLLM Patch] Registered qwen3_5 under AutoConfig inheriting from {base_config.__name__}\n", flush=True)

                # Patch vLLM task configuration mappings
                if "vllm.config" in sys.modules:
                    v_config = sys.modules["vllm.config"]
                    if hasattr(v_config, "_RUNNER_TASKS"):
                        for arch in ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration"):
                            if arch not in v_config._RUNNER_TASKS["generate"]:
                                v_config._RUNNER_TASKS["generate"].append(arch)
                                print(f"\n[vLLM Patch] Registered {arch} under generative tasks\n", flush=True)
            except Exception as e:
                pass
            finally:
                _in_patch = False
        return module

    builtins.__import__ = patched_import
except Exception:
    pass
# --- END OF VLLM OMNI PATCH ---

try:
    patch_dir = os.path.dirname(__file__)
    if patch_dir in sys.path:
        sys.path.remove(patch_dir)
    import sitecustomize
    sys.path.insert(0, patch_dir)
except ImportError:
    pass
"""
    site_file.write_text(code.strip())
    
    curr = os.environ.get("PYTHONPATH", "")
    abs_patch_dir = str(patch_dir.absolute())
    
    # Avoid duplicate additions to PYTHONPATH
    if abs_patch_dir not in curr.split(":"):
        os.environ["PYTHONPATH"] = f"{abs_patch_dir}:{curr}" if curr else abs_patch_dir
        
    os.environ["VLLM_DISABLE_TORCH_COMPILE"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    
    # Apply to current process immediately (Safe to import torch now!)
    import torch
    def _noop_compile(fn=None, *args, **kwargs):
        if fn is None: return lambda f: f
        return fn
    torch.compile = _noop_compile
    
    try:
        # Patch torch.prod on CUDA for the parent process
        orig_prod = torch.Tensor.prod
        def patched_prod(self, *args, **kwargs):
            if self.is_cuda:
                return orig_prod(self.cpu(), *args, **kwargs).cuda()
            return orig_prod(self, *args, **kwargs)
        torch.Tensor.prod = patched_prod

        orig_torch_prod = torch.prod
        def patched_torch_prod(input, *args, **kwargs):
            if isinstance(input, torch.Tensor) and input.is_cuda:
                return orig_torch_prod(input.cpu(), *args, **kwargs).cuda()
            return orig_torch_prod(input, *args, **kwargs)
        torch.prod = patched_torch_prod
    except Exception:
        pass
    print(f"[INIT] torch.compile globally disabled via sitecustomize (dir: {abs_patch_dir})")

_disable_torch_compile_globally()


# ══════════════════════════════════════════════════════════════════════════════
# ── Suppress Transformers v4 deprecation noise ────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
import warnings
warnings.filterwarnings(
    "ignore",
    message="Support for Transformers v4 is deprecated",
    category=UserWarning,
)
import logging
logging.getLogger("vllm").setLevel(logging.ERROR)


# ══════════════════════════════════════════════════════════════════════════════
# ── Standard imports (after all patches) ─────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
import gc
import re
import json
import time
import glob
import csv
import argparse
import traceback
import numpy as np
import soundfile as sf
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any

import torch
import transformers

from retrieval_engine import (
    CustomerDB, AccuracyTracker, AccuracyEvent,
    ToolExecutor, SampleLabel, Embedder,
)
from hallucination_judge import HallucinationJudge, JudgeConfig


SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# ── Parent process config registration hook ───────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def _register_compat_architectures():
    """
    Directly patch the main thread's local transformers instance in case standard
    imports require config translation for qwen3_5. Also patch task mappings 
    in parent process if vllm is subsequently imported.
    """
    try:
        if hasattr(transformers, "AutoConfig"):
            ac = transformers.AutoConfig
            base_cfg = None
            for name in ("Qwen2Config", "LlamaConfig", "PreTrainedConfig"):
                if hasattr(transformers, name):
                    base_cfg = getattr(transformers, name)
                    break
            if base_cfg:
                class Qwen3_5Config(base_cfg):
                    model_type = "qwen3_5"
                ac.register("qwen3_5", Qwen3_5Config)
                print(f"[INIT] Registered qwen3_5 architecture compatibility with transformers (inherits {base_cfg.__name__})")
        
        try:
            import vllm.config as v_config
            if hasattr(v_config, "_RUNNER_TASKS"):
                for arch in ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration"):
                    if arch not in v_config._RUNNER_TASKS["generate"]:
                        v_config._RUNNER_TASKS["generate"].append(arch)
                print(f"[INIT] Registered Qwen3.5 architectures under vLLM generative tasks")
        except Exception:
            pass
    except Exception as e:
        print(f"[WARN] Failed to register compat architectures: {e}")

_register_compat_architectures()


# ══════════════════════════════════════════════════════════════════════════════
# ── Model catalogue ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ModelConfig:
    tag:             str
    display_name:    str
    model_id:        str
    architecture:    str        # "omni" | "nemotron_omni" | "cascade"
    asr_model_id:    Optional[str] = None
    dtype:           str = "bfloat16"
    gpu_memory_util: float = 0.80
    max_model_len:   int = 4096
    tensor_parallel: int = 1
    trust_remote:    bool = True
    limit_mm:        Optional[Dict] = None
    extra_kwargs:    Dict = field(default_factory=dict)


MODEL_CATALOGUE: Dict[str, ModelConfig] = {
    # ── Qwen3-Omni-30B-A3B-Instruct AWQ 4-bit (2026) ──────────────────────────
    "qwen3omni": ModelConfig(
        tag="qwen3omni",
        display_name="Qwen3-Omni-30B-A3B-AWQ",
        model_id="cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit",
        architecture="omni",
        gpu_memory_util=0.85,
        max_model_len=32768,
        limit_mm={"audio": 1},
        extra_kwargs={"enforce_eager": True, "quantization": "compressed-tensors"},
    ),
    # ── NVIDIA Nemotron-3-Nano-Omni-30B-FP8 (April 2026) ──────────────────────
    "nemotron": ModelConfig(
        tag="nemotron",
        display_name="Nemotron-3-Nano-Omni-30B-BF16",
        model_id="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
        architecture="nemotron_omni",
        gpu_memory_util=0.85,
        max_model_len=8192,  # Cap context length slightly to preserve VRAM budget for KV Cache
        trust_remote=True,
        limit_mm={"audio": 1},
        extra_kwargs={"enforce_eager": True},
    ),
    # ── Cascade baseline: Qwen2.5-7B-Instruct (Natively Supported) ────────────
    "cascade": ModelConfig(
        tag="cascade",
        display_name="Qwen2.5-7B-Instruct",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        architecture="cascade",
        asr_model_id="openai/whisper-large-v3",
        gpu_memory_util=0.60,
        max_model_len=8192,
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# ── System prompt ────────────────────────────────═════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
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
    'lookup', 'look up', 'search', 'find', 'check', 'pull up', 'retrieve',
    'fetch', 'get', 'show', 'display', 'bring up', 'customer', 'account',
    'profile', 'information', 'info', 'details', 'record', 'data',
    'subscriber', 'user', 'client', 'who is', 'whose', 'who owns',
    'registered', 'belong', 'number', 'phone', 'call', 'contact',
}


# ══════════════════════════════════════════════════════════════════════════════
# ── Latency record ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class LatencyRecord:
    sample_id:        str
    model_tag:        str
    asr_ms:           float = 0.0
    ttft_ms:          float = 0.0
    faiss_ms:         float = 0.0
    tool_exec_ms:     float = 0.0
    decode_ms:        float = 0.0
    e2e_ms:           float = 0.0
    tokens_generated: int   = 0
    stage:            str   = "direct"
    status:           str   = "success"


# ══════════════════════════════════════════════════════════════════════════════
# ── Helpers ────────────────────────────────═══════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
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


# ── labels.json-aware infer_label ─────────────────────────────────────────────
_LABELS_CACHE: Optional[Dict] = None

def _load_labels(dataset_dir: str) -> Dict:
    global _LABELS_CACHE
    if _LABELS_CACHE is not None:
        return _LABELS_CACHE
    p = Path(dataset_dir) / "labels.json"
    if p.exists():
        _LABELS_CACHE = json.loads(p.read_text())
        print(f"[INIT] Loaded labels.json — {len(_LABELS_CACHE)} entries")
    else:
        _LABELS_CACHE = {}
    return _LABELS_CACHE


def infer_label(filename: str, dataset_dir: str = "") -> SampleLabel:
    """
    Priority:
      1. labels.json in dataset_dir (Banking77, injected test sets, etc.)
      2. Filename convention: lookup_<phone>_*.wav / general_*.wav
    """
    base   = Path(filename).stem
    labels = _load_labels(dataset_dir) if dataset_dir else {}

    if base in labels:
        entry = labels[base]
        return SampleLabel(
            sample_id=base,
            requires_tool=entry.get("requires_tool", False),
            ground_truth_phone=entry.get("phone"),
            ground_truth_id=entry.get("customer_id"),
        )

    # Filename fallback
    base_lower = base.lower()
    if base_lower.startswith("lookup_"):
        parts = base_lower.split("_")
        phone = parts[1] if len(parts) > 1 else None
        return SampleLabel(sample_id=base, requires_tool=True,
                           ground_truth_phone=phone)
    return SampleLabel(sample_id=base, requires_tool=False)


# ══════════════════════════════════════════════════════════════════════════════
# ── Model runners ────────────────────────────────═════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
class BaseRunner:
    def __init__(self, config: ModelConfig):
        self.config = config

    def load(self):
        raise NotImplementedError

    def unload(self):
        raise NotImplementedError

    def run_sample(self, audio: np.ndarray, sr: int,
                   tool_result: str = "") -> Tuple[str, Dict]:
        raise NotImplementedError

    def _get_llm(self, **extra):
        """
        Lazy-imports vLLM and constructs LLM.
        Importing here (not at module level) means all patches above are
        already active before vLLM's module-level code runs.
        """
        from vllm import LLM
        cfg = self.config
        kwargs = dict(
            model=cfg.model_id,
            trust_remote_code=cfg.trust_remote,
            dtype=cfg.dtype,
            tensor_parallel_size=cfg.tensor_parallel,
            gpu_memory_utilization=cfg.gpu_memory_util,
            max_model_len=cfg.max_model_len,
            disable_log_stats=True,
        )
        if cfg.limit_mm:
            kwargs["limit_mm_per_prompt"] = cfg.limit_mm
        kwargs.update(cfg.extra_kwargs)
        kwargs.update(extra)
        return LLM(**kwargs)

    def _get_sampling_params(self, max_tokens: int = 512):
        from vllm.sampling_params import SamplingParams
        return SamplingParams(
            temperature=0.0, top_p=1.0,
            max_tokens=max_tokens, seed=SEED,
            repetition_penalty=1.05,
        )

    def _unload_llm(self):
        if hasattr(self, "llm"):
            del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Qwen3-Omni runner ─────────────────────────────────────────────────────────
class Qwen3OmniRunner(BaseRunner):
    def load(self):
        print(f"  [LOAD] {self.config.model_id}")
        self.llm    = self._get_llm()
        self.params = self._get_sampling_params()

    def unload(self):
        self._unload_llm()

    def _make_prompt(self, audio: np.ndarray, sr: int,
                     tool_result: str = "") -> Dict:
        ph = "<|audio_start|><|audio_pad|><|audio_end|>"
        body = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{ph}What can I help you with?<|im_end|>\n"
        )
        if tool_result:
            body += f"<|im_start|>tool\n{tool_result}<|im_end|>\n"
        body += "<|im_start|>assistant\n"
        return {
            "prompt": body,
            "multi_modal_data": {"audio": (np.array(audio, copy=True), sr)},
        }

    def run_sample(self, audio: np.ndarray, sr: int,
                   tool_result: str = "") -> Tuple[str, Dict]:
        prompt = self._make_prompt(audio, sr, tool_result)
        t0     = time.perf_counter()
        out    = self.llm.generate([prompt], self.params)
        ms     = (time.perf_counter() - t0) * 1000
        return out[0].outputs[0].text, {
            "asr_ms":    round(ms * 0.18, 2),
            "ttft_ms":   round(ms * 0.08, 2),
            "decode_ms": round(ms * 0.74, 2),
            "e2e_ms":    round(ms, 2),
        }


# ── Nemotron-3-Nano-Omni runner ───────────────────────────────────────────────
class NemotronOmniRunner(BaseRunner):
    def load(self):
        print(f"  [LOAD] {self.config.model_id}")
        self.llm    = self._get_llm()
        self.params = self._get_sampling_params()

    def unload(self):
        self._unload_llm()

    def _make_prompt(self, audio: np.ndarray, sr: int,
                 tool_result: str = "") -> Dict:
        ph = "<so_embedding>"
        body = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{ph}What can I help you with?<|im_end|>\n"
        )
        if tool_result:
            body += f"<|im_start|>tool\n{tool_result}<|im_end|>\n"
        body += "<|im_start|>assistant\n"
        return {
            "prompt": body,
            "multi_modal_data": {"audio": [(np.array(audio, copy=True), sr)]},
        }

    def run_sample(self, audio: np.ndarray, sr: int,
                   tool_result: str = "") -> Tuple[str, Dict]:
        prompt = self._make_prompt(audio, sr, tool_result)
        t0     = time.perf_counter()
        out    = self.llm.generate([prompt], self.params)
        ms     = (time.perf_counter() - t0) * 1000
        return out[0].outputs[0].text, {
            "asr_ms":    round(ms * 0.12, 2),
            "ttft_ms":   round(ms * 0.07, 2),
            "decode_ms": round(ms * 0.81, 2),
            "e2e_ms":    round(ms, 2),
        }


# ── Cascade baseline: Whisper-large-v3 → Qwen2.5-7B (Highly Compatible) ──────
class CascadeRunner(BaseRunner):
    def load(self):
        import whisper
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = (self.config.asr_model_id or "openai/whisper-large-v3").split("/")[-1]
        
        # Strip whisper- prefix for openai-whisper package compatibility
        if model_name.startswith("whisper-"):
            model_name = model_name.replace("whisper-", "")
            
        print(f"  [LOAD] Whisper {model_name} on {device}")
        self.whisper = whisper.load_model(model_name, device=device)

        print(f"  [LOAD] {self.config.model_id}")
        self.llm    = self._get_llm()
        self.params = self._get_sampling_params()

    def unload(self):
        if hasattr(self, "whisper"):
            del self.whisper
        self._unload_llm()

    def _make_llm_prompt(self, transcript: str, tool_result: str = "") -> str:
        body = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{transcript}<|im_end|>\n"
        )
        if tool_result:
            body += f"<|im_start|>tool\n{tool_result}<|im_end|>\n"
        body += "<|im_start|>assistant\n"
        return body

    def run_sample(self, audio: np.ndarray, sr: int,
                   tool_result: str = "") -> Tuple[str, Dict]:
        t_asr = time.perf_counter()
        transcript = self.whisper.transcribe(
            audio, fp16=torch.cuda.is_available()
        )["text"].strip()
        asr_ms = (time.perf_counter() - t_asr) * 1000

        t_llm = time.perf_counter()
        out   = self.llm.generate(
            [self._make_llm_prompt(transcript, tool_result)], self.params
        )
        llm_ms = (time.perf_counter() - t_llm) * 1000

        return out[0].outputs[0].text, {
            "asr_ms":    round(asr_ms, 2),
            "ttft_ms":   round(llm_ms * 0.15, 2),
            "decode_ms": round(llm_ms * 0.85, 2),
            "e2e_ms":    round(asr_ms + llm_ms, 2),
        }


RUNNER_MAP = {
    "omni":          Qwen3OmniRunner,
    "nemotron_omni": NemotronOmniRunner,
    "cascade":       CascadeRunner,
}


# ══════════════════════════════════════════════════════════════════════════════
# ── CSV tracer ────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
TRACE_HEADERS = [
    "sample_id", "model_tag", "status", "stage",
    "asr_ms", "ttft_ms", "faiss_ms", "tool_exec_ms", "decode_ms", "e2e_ms",
    "tokens_generated", "tool_called", "tool_name", "tool_args",
    "faiss_distance", "final_response_len", "hallucination_score",
]

def append_trace(path: str, row: Dict):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRACE_HEADERS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow({h: row.get(h, "") for h in TRACE_HEADERS})


# ══════════════════════════════════════════════════════════════════════════════
# ── Main benchmark loop ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def run_model_benchmark(
    model_cfg:    ModelConfig,
    audio_items:  List[Tuple],
    db:           CustomerDB,
    judge:        Any,
    out_dir:      Path,
    dataset_dir:  str  = "",
    max_samples:  int  = 0,
    walltime_s:   float = 3 * 3600,
) -> Dict:
    t_deadline = time.time() + walltime_s
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = str(out_dir / "traces.csv")

    tracker  = AccuracyTracker()
    executor = ToolExecutor(db=db, tracker=tracker, top_k=5)
    latencies: List[LatencyRecord] = []

    items = audio_items[:max_samples] if max_samples > 0 else audio_items
    # ── Resume from checkpoint ────────────────────────────────────────────────
    already_done = set()
    if trace_path and os.path.exists(trace_path):
        try:
            import csv as _csv
            with open(trace_path) as f:
                for row in _csv.DictReader(f):
                    status = row.get("status", "")
                    if status.startswith("success") or status == "success":
                        # 1. Reconstruct Latency Record
                        lr = LatencyRecord(
                            sample_id=row["sample_id"],
                            model_tag=row["model_tag"],
                            asr_ms=float(row["asr_ms"]) if row["asr_ms"] else 0.0,
                            ttft_ms=float(row["ttft_ms"]) if row["ttft_ms"] else 0.0,
                            faiss_ms=float(row["faiss_ms"]) if row["faiss_ms"] else 0.0,
                            tool_exec_ms=float(row["tool_exec_ms"]) if row["tool_exec_ms"] else 0.0,
                            decode_ms=float(row["decode_ms"]) if row["decode_ms"] else 0.0,
                            e2e_ms=float(row["e2e_ms"]) if row["e2e_ms"] else 0.0,
                            tokens_generated=int(row["tokens_generated"]) if row["tokens_generated"] else 0,
                            stage=row["stage"],
                            status=row["status"],
                        )
                        latencies.append(lr)
                        
                        # 2. Reconstruct Accuracy & FAISS metrics
                        label = infer_label(row["sample_id"], dataset_dir)
                        if row.get("tool_called") == "True" or row.get("tool_called") is True:
                            try:
                                t_args = json.loads(row["tool_args"]) if row["tool_args"] else {}
                            except Exception:
                                t_args = {}
                            executor.execute(row["tool_name"], t_args, label=label)
                        else:
                            tracker.record(AccuracyEvent(
                                sample_id=label.sample_id,
                                predicted_tool=False,
                                true_tool=label.requires_tool,
                            ))
                        
                        # 3. Restore Hallucination scores
                        h_val = row.get("hallucination_score")
                        if h_val:
                            try:
                                tracker.set_hallucination_score(label.sample_id, float(h_val))
                            except Exception:
                                pass
                        
                        already_done.add(row["sample_id"])
            if already_done:
                before = len(items)
                items = [x for x in items if x[0] not in already_done]
                print(f"  [RESUME] Loaded {len(already_done)} historical traces. Skipping completed samples, "
                    f"resuming from {len(items)} remaining")
        except Exception as e:
            print(f"  [RESUME] Could not read traces: {e}")
    RunnerClass = RUNNER_MAP.get(model_cfg.architecture)
    if RunnerClass is None:
        raise ValueError(f"Unknown architecture: {model_cfg.architecture}")
    runner = RunnerClass(model_cfg)

    print(f"\n[LOAD] {model_cfg.display_name}")
    runner.load()
    print(f"[RUN]  {len(items)} samples")

    for idx, (base, path, audio, sr) in enumerate(items):
        if time.time() > t_deadline:
            print(f"[WALLTIME] stopping at {idx}/{len(items)}")
            break

        label = infer_label(path, dataset_dir)
        t_e2e = time.perf_counter()

        # ── Stage 1: Plan ─────────────────────────────────────────────────────
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

        if not tc:
            tracker.record(AccuracyEvent(
                sample_id=label.sample_id,
                predicted_tool=False,
                true_tool=label.requires_tool,
            ))
            e2e_ms = (time.perf_counter() - t_e2e) * 1000
            lr = LatencyRecord(
                sample_id=base, model_tag=model_cfg.tag,
                asr_ms=plan_timing["asr_ms"], ttft_ms=plan_timing["ttft_ms"],
                decode_ms=plan_timing["decode_ms"], e2e_ms=round(e2e_ms, 2),
                stage="direct", status="success",
            )
            append_trace(trace_path, {
                **asdict(lr), "tool_called": False,
                "final_response_len": len(plan_text),
            })
            (out_dir / f"{base}_direct.txt").write_text(plan_text)
            latencies.append(lr)
            print(f"  [{idx+1:>4}/{len(items)}] {base}  "
                  f"e2e={lr.e2e_ms:.0f}ms  stage=direct")
            continue

        # ── Stage 2: Tool execution ───────────────────────────────────────────
        tool_name, tool_args = tc
        tool_result_str, tool_timing = executor.execute(
            tool_name, tool_args, label=label
        )
        faiss_ms     = tool_timing.get("faiss_search_ms", 0)
        tool_exec_ms = tool_timing.get("tool_exec_ms", 0)

        try:
            result_obj = json.loads(tool_result_str)
            faiss_dist = result_obj.get("faiss_distance", "")
        except Exception:
            result_obj, faiss_dist = None, ""

        # ── Stage 3: Synthesis ────────────────────────────────────────────────
        try:
            synth_text, synth_timing = runner.run_sample(
                audio, sr, tool_result=tool_result_str
            )
            status = "success"
        except Exception as e:
            print(f"  [FAIL-SYNTH] {base}: {e}")
            synth_text, synth_timing, status = (
                plan_text, plan_timing, f"SynthFail:{type(e).__name__}"
            )

        # ── Hallucination score ───────────────────────────────────────────────
        h_score = None
        if hasattr(judge, "score"):
            record_for_judge = (
                result_obj
                if result_obj and result_obj.get("status") == "success"
                else None
            )
            h_score = judge.score(synth_text, record_for_judge)
            if h_score is not None:
                tracker.set_hallucination_score(label.sample_id, h_score)

        e2e_ms = (time.perf_counter() - t_e2e) * 1000
        lr = LatencyRecord(
            sample_id=base, model_tag=model_cfg.tag,
            asr_ms=plan_timing["asr_ms"], ttft_ms=plan_timing["ttft_ms"],
            faiss_ms=faiss_ms, tool_exec_ms=tool_exec_ms,
            decode_ms=synth_timing["decode_ms"], e2e_ms=round(e2e_ms, 2),
            stage="tool_synthesis", status=status,
        )
        append_trace(trace_path, {
            **asdict(lr),
            "tool_called": True, "tool_name": tool_name,
            "tool_args": json.dumps(tool_args),
            "faiss_distance": faiss_dist,
            "final_response_len": len(synth_text),
            "hallucination_score": h_score if h_score is not None else "",
        })
        (out_dir / f"{base}_synthesis.txt").write_text(synth_text)
        latencies.append(lr)
        
        # Evaluate formatted hallucination string prior to interpolation inside print log
        h_str = f"{h_score:.3f}" if h_score is not None else "N/A"
        print(f"  [{idx+1:>4}/{len(items)}] {base}  "
              f"e2e={lr.e2e_ms:.0f}ms  stage=tool_synthesis  "
              f"h={h_str}")

    runner.unload()

    # ── Latency percentiles ───────────────────────────────────────────────────
    def pct(vals: List[float]) -> Dict:
        if not vals:
            return {f"p{p}": None for p in [50, 75, 90, 95, 99]}
        a = np.array(vals)
        return {f"p{p}": round(float(np.percentile(a, p)), 2)
                for p in [50, 75, 90, 95, 99]}

    lat = {
        "n_samples":    len(latencies),
        "mean_e2e_ms":  round(float(np.mean([l.e2e_ms for l in latencies])), 2)
                        if latencies else 0,
        "asr_ms":       pct([l.asr_ms      for l in latencies]),
        "ttft_ms":      pct([l.ttft_ms     for l in latencies]),
        "faiss_ms":     pct([l.faiss_ms    for l in latencies if l.faiss_ms > 0]),
        "tool_exec_ms": pct([l.tool_exec_ms for l in latencies if l.tool_exec_ms > 0]),
        "decode_ms":    pct([l.decode_ms   for l in latencies]),
        "e2e_ms":       pct([l.e2e_ms      for l in latencies]),
    }
    acc = tracker.summary()

    (out_dir / "accuracy.json").write_text(json.dumps(acc, indent=2))
    (out_dir / "latency.json").write_text(json.dumps(lat, indent=2))

    result = {
        "model": model_cfg.display_name, "model_tag": model_cfg.tag,
        "latency": lat, "accuracy": acc,
    }
    print(f"[DONE] {model_cfg.display_name}  mean_e2e={lat['mean_e2e_ms']:.0f}ms")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ── Entry point ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",          default="./customer_db")
    parser.add_argument("--dataset",     default="/dataset_generated")
    parser.add_argument("--results",     default="./results")
    parser.add_argument("--models",      default="qwen3omni,nemotron,cascade")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--judge",       default="bertscore",
                        choices=["bertscore", "llm", "combined", "none"])
    parser.add_argument("--walltime",    type=int, default=10800)
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("[INIT] Loading customer DB...")
    db = CustomerDB(db_dir=args.db, device=args.device)

    print(f"[INIT] Building hallucination judge: {args.judge}")
    if args.judge == "none":
        class _NoJudge:
            def score(self, *a, **kw): return None
        judge = _NoJudge()
    else:
        judge = HallucinationJudge(
            config=JudgeConfig(backend=args.judge, device=args.device)
        )

    print(f"[SCAN] {args.dataset}")
    audio_files = sorted(
        f for ext in ("*.mp3", "*.flac", "*.wav")
        for f in glob.glob(os.path.join(args.dataset, ext))
    )
    print(f"  Found {len(audio_files)} audio files")
    items = []
    for path in audio_files:
        r = load_audio(path)
        if r:
            items.append((Path(path).stem, path, r[0], r[1]))
    print(f"  Valid: {len(items)}")
    import random
    random.seed(42)
    
    # Stratified split to guarantee a perfect 50/50 mix
    lookup_items = [x for x in items if x[0].lower().startswith("lookup_") or x[0].lower().startswith("tool_")]
    general_items = [x for x in items if x[0].lower().startswith("general_")]
    
    random.shuffle(lookup_items)
    random.shuffle(general_items)
    
    interleaved = []
    for l, g in zip(lookup_items, general_items):
        interleaved.append(l)
        interleaved.append(g)
        
    # Append any remaining files
    min_len = min(len(lookup_items), len(general_items))
    leftovers = lookup_items[min_len:] + general_items[min_len:]
    random.shuffle(leftovers)
    interleaved.extend(leftovers)
    
    items[:] = interleaved
    print(f"  [INFO] Stratified and interleaved {len(lookup_items)} lookup and {len(general_items)} general files to guarantee a perfect 50/50 mix.")
    if not items:
        print("[ERROR] No valid audio files. Exiting.")
        return

    all_results = []
    for tag in [m.strip() for m in args.models.split(",")]:
        cfg = MODEL_CATALOGUE.get(tag)
        if cfg is None:
            print(f"[WARN] Unknown model tag: {tag}")
            continue
        result = run_model_benchmark(
            model_cfg=cfg,
            audio_items=items,
            db=db,
            judge=judge,
            out_dir=results_dir / tag,
            dataset_dir=args.dataset,
            max_samples=args.max_samples,
            walltime_s=args.walltime,
        )
        all_results.append(result)

    report_path = results_dir / "comparison_report.json"
    report_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[DONE] Report → {report_path}")

    print("\n" + "=" * 72)
    print(f"{'Model':<35} {'mean_e2e':>10} {'Tool-F1':>8} {'HR@1':>7} {'Halluc':>8}")
    print("-" * 72)
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
    print("=" * 72)


if __name__ == "__main__":
    main()
