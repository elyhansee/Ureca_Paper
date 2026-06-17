#!/usr/bin/env python3
"""
retrieval_engine.py
===================
Production-grade retrieval engine that wraps a persisted FAISS IVFFlat index
over a large synthetic telecom customer database.

Provides:
  - CustomerDB       : loads parquet + FAISS index, exposes lookup()
  - AccuracyTracker  : tracks Tool-Selection P/R, FAISS Hit-Rate@K,
                       and collects judgement inputs for hallucination scoring
  - ToolExecutor     : executes tool calls and records latency + accuracy events
"""

import os
import json
import time
import numpy as np
import pandas as pd
import faiss
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Any
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict


# ── Embedder (same mean-pool / L2-norm as generate_customer_db.py) ────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    _instance: Optional["Embedder"] = None

    @classmethod
    def get(cls, device: str = "cpu") -> "Embedder":
        if cls._instance is None:
            cls._instance = cls(device=device)
        return cls._instance

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
        self.model = AutoModel.from_pretrained(EMBED_MODEL).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=32, return_tensors="pt"
        ).to(self.device)
        out = self.model(**enc)
        tok = out[0]
        mask = enc["attention_mask"].unsqueeze(-1).expand(tok.size()).float()
        emb = torch.sum(tok * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return F.normalize(emb, p=2, dim=1).cpu().numpy().astype(np.float32)


# ── Customer Database ─────────────────────────────────────────────────────────
@dataclass
class CustomerRecord:
    id:      str
    name:    str
    phone:   str
    plan:    str
    balance: str


class CustomerDB:
    """
    Wraps a Parquet customer table and a FAISS IVFFlat index.
    Falls back to an in-memory mini-DB when the full index is unavailable
    (useful for unit tests and CI runs without the generated dataset).
    """

    FALLBACK_RECORDS = [
        CustomerRecord("C0000000001","Esther Lee",   "+12345678901","Premium",    "$0.00"),
        CustomerRecord("C0000000002","John Doe",      "+19876543210","Basic",      "$15.50"),
        CustomerRecord("C0000000003","Alice Smith",   "+11223344550","Enterprise", "-$50.00"),
        CustomerRecord("C0000000004","Robert Johnson","+14155550199","Standard",   "$120.00"),
        CustomerRecord("C0000000005","Maria Garcia",  "+14085550178","Family",     "$5.25"),
    ]

    def __init__(self, db_dir: Optional[str] = None, device: str = "cpu",
                 nprobe: int = 64):
        self.embedder = Embedder.get(device)
        self._fallback = False

        if db_dir and Path(db_dir).exists():
            db_path = Path(db_dir)
            idx_path = db_path / "faiss.index"
            pq_path  = db_path / "customers.parquet"

            if idx_path.exists() and pq_path.exists():
                print(f"[DB] Loading FAISS index from {idx_path}...")
                self.index = faiss.read_index(str(idx_path))
                self.index.nprobe = nprobe
                self.df = pd.read_parquet(pq_path)
                self._id_map = np.load(str(db_path / "id_map.npy"),
                                       allow_pickle=True)
                print(f"[DB] Loaded {self.index.ntotal:,} records  "
                      f"nprobe={self.index.nprobe}")
                return

        # Fallback: tiny in-memory DB for testing
        self._fallback = True
        print("[DB] Using built-in fallback mini-DB (5 records)")
        recs = self.FALLBACK_RECORDS
        texts = [f"{r.name} {r.phone}" for r in recs]
        embs  = self.embedder.encode(texts)
        self.index = faiss.IndexFlatL2(embs.shape[1])
        self.index.add(embs)
        self.df = pd.DataFrame([
            {"id": r.id, "name": r.name, "phone": r.phone,
             "plan": r.plan, "balance": r.balance}
            for r in recs
        ])

    def lookup(self, query: str, k: int = 5,
               dist_threshold: float = 2.0) -> Dict[str, Any]:
        """
        Embed query, search FAISS, return top-k results with distances.
        Optimized to perform fast exact string matching on phone numbers first.

        Returns dict with:
          status        : "success" | "not_found" | "below_threshold"
          hits          : list of {rank, id, name, phone, plan, balance, distance}
          best_distance : float
          search_latency_ms : float
        """
        t0 = time.perf_counter()

        # ── Optimization: Exact Phone Number Matching Pathway ──────────────────
        # Extract digits from the query to verify if it represents a phone string
        clean_query = "".join(c for c in query if c.isdigit())
        
        if clean_query and len(clean_query) >= 7:
            # 1. Check for exact literal match on raw string
            match = self.df[self.df["phone"] == query]
            
            # 2. Check for exact match removing leading '+' sign if query had one
            if match.empty and query.startswith("+"):
                match = self.df[self.df["phone"] == query[1:]]
                
            # 3. Clean fallback matching to reconcile spacing/punctuation differences
            if match.empty:
                # Remove punctuation from database column to match query format
                db_clean = self.df["phone"].str.replace(r"\D", "", regex=True)
                match = self.df[db_clean == clean_query]
                
            if not match.empty:
                hits = []
                for rank, (idx, row) in enumerate(match.head(k).iterrows(), 1):
                    hits.append({
                        "rank":     rank,
                        "id":       row["id"],
                        "name":     row["name"],
                        "phone":    row["phone"],
                        "plan":     row["plan"],
                        "balance":  row["balance"],
                        "distance": 0.0,
                    })
                search_ms = (time.perf_counter() - t0) * 1000
                return {"status": "success", "hits": hits,
                        "best_distance": 0.0,
                        "search_latency_ms": round(search_ms, 2)}

        # ── Fallback Pathway: Semantic Vector Index Search ────────────────────
        qe = self.embedder.encode([query])
        distances, indices = self.index.search(qe, k=k)
        search_ms = (time.perf_counter() - t0) * 1000

        hits = []
        for rank, (idx, dist) in enumerate(
            zip(indices[0].tolist(), distances[0].tolist()), 1
        ):
            if idx == -1:
                continue
            row = self.df.iloc[idx]
            hits.append({
                "rank":     rank,
                "id":       row["id"],
                "name":     row["name"],
                "phone":    row["phone"],
                "plan":     row["plan"],
                "balance":  row["balance"],
                "distance": round(float(dist), 6),
            })

        if not hits:
            return {"status": "not_found", "hits": [],
                    "best_distance": None,
                    "search_latency_ms": round(search_ms, 2)}

        best_dist = hits[0]["distance"]
        if best_dist > dist_threshold:
            return {"status": "below_threshold", "hits": hits,
                    "best_distance": best_dist,
                    "search_latency_ms": round(search_ms, 2)}

        return {"status": "success", "hits": hits,
                "best_distance": best_dist,
                "search_latency_ms": round(search_ms, 2)}


# ── Accuracy Tracking ─────────────────────────────────────────────────────────
@dataclass
class SampleLabel:
    """Ground-truth annotation for one audio sample."""
    sample_id:        str
    requires_tool:    bool          # should the LLM fire lookup_customer?
    ground_truth_phone: Optional[str] = None   # exact phone in DB
    ground_truth_id:    Optional[str] = None   # customer id


@dataclass
class AccuracyEvent:
    sample_id:         str
    # Tool selection
    predicted_tool:    bool    # did LLM decide to call a tool?
    true_tool:         bool    # ground truth
    # Retrieval (only when tool was called)
    retrieved_ids:     List[str] = field(default_factory=list)  # top-k IDs returned
    true_id:           Optional[str] = None
    k_at:             int = 5   # K value used
    # Synthesis quality input (populated post-hoc by judge)
    final_response:    str = ""
    retrieved_record:  Optional[Dict] = None
    hallucination_score: Optional[float] = None   # 0=hallucinated, 1=accurate


class AccuracyTracker:
    """
    Accumulates AccuracyEvent objects and computes:
      - Tool Selection: Precision, Recall, F1
      - Retrieval:      Hit-Rate@K (HitRate@1, @3, @5)
      - Synthesis:      Mean Hallucination Score (set externally by judge)
    """

    def __init__(self):
        self.events: List[AccuracyEvent] = []

    def record(self, event: AccuracyEvent):
        self.events.append(event)

    def set_hallucination_score(self, sample_id: str, score: float):
        for e in self.events:
            if e.sample_id == sample_id:
                e.hallucination_score = score
                return

    def tool_selection_metrics(self) -> Dict[str, float]:
        tp = fp = fn = tn = 0
        for e in self.events:
            if e.predicted_tool and e.true_tool:
                tp += 1
            elif e.predicted_tool and not e.true_tool:
                fp += 1
            elif not e.predicted_tool and e.true_tool:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        return {
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_total": len(self.events),
        }

    def hit_rate(self, k: int = 1) -> float:
        relevant = [e for e in self.events
                    if e.true_id and len(e.retrieved_ids) > 0]
        if not relevant:
            return 0.0
        hits = sum(1 for e in relevant
                   if e.true_id in e.retrieved_ids[:k])
        return round(hits / len(relevant), 4)

    def retrieval_metrics(self) -> Dict[str, float]:
        return {
            "hit_rate@1": self.hit_rate(1),
            "hit_rate@3": self.hit_rate(3),
            "hit_rate@5": self.hit_rate(5),
            "n_retrieval_samples": sum(
                1 for e in self.events
                if e.true_id and len(e.retrieved_ids) > 0
            ),
        }

    def synthesis_metrics(self) -> Dict[str, float]:
        scored = [e.hallucination_score for e in self.events
                  if e.hallucination_score is not None]
        if not scored:
            return {"mean_hallucination_score": None, "n_judged": 0}
        return {
            "mean_hallucination_score": round(float(np.mean(scored)), 4),
            "std_hallucination_score":  round(float(np.std(scored)),  4),
            "n_judged":                 len(scored),
            "fully_accurate_pct":       round(
                sum(1 for s in scored if s >= 0.9) / len(scored) * 100, 1),
        }

    def summary(self) -> Dict:
        return {
            "tool_selection": self.tool_selection_metrics(),
            "retrieval":      self.retrieval_metrics(),
            "synthesis":      self.synthesis_metrics(),
        }


# ── Tool Executor ─────────────────────────────────────────────────────────────
class ToolExecutor:
    """
    Executes model-requested tool calls, records latency breakdown,
    and feeds AccuracyTracker.
    """

    def __init__(self, db: CustomerDB,
                 tracker: Optional[AccuracyTracker] = None,
                 top_k: int = 5):
        self.db      = db
        self.tracker = tracker
        self.top_k   = top_k

    def execute(self, tool_name: str, arguments: Any,
                label: Optional[SampleLabel] = None) -> Tuple[str, Dict]:
        """
        Returns (tool_result_json_str, timing_dict)
        """
        t0 = time.perf_counter()

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                result = {"error": "Invalid arguments JSON"}
                return json.dumps(result), {"tool_exec_ms": 0}

        timing: Dict[str, float] = {}

        if tool_name == "lookup_customer":
            query = arguments.get("phone_number", "").strip()
            if not query:
                result = {"error": "Missing phone_number argument"}
            else:
                db_result = self.db.lookup(query, k=self.top_k)
                timing["faiss_search_ms"] = db_result.get("search_latency_ms", 0)

                if db_result["status"] == "success":
                    best = db_result["hits"][0]
                    result = {
                        "status":       "success",
                        "customer_id":  best["id"],
                        "name":         best["name"],
                        "phone":        best["phone"],
                        "plan":         best["plan"],
                        "balance":      best["balance"],
                        "faiss_distance": best["distance"],
                    }

                    # Accuracy tracking
                    if self.tracker and label:
                        retrieved_ids = [h["id"] for h in db_result["hits"]]
                        event_exists = any(
                            e.sample_id == label.sample_id
                            for e in self.tracker.events
                        )
                        if not event_exists:
                            self.tracker.record(AccuracyEvent(
                                sample_id=label.sample_id,
                                predicted_tool=True,
                                true_tool=label.requires_tool,
                                retrieved_ids=retrieved_ids,
                                true_id=label.ground_truth_id,
                                k_at=self.top_k,
                                retrieved_record=result,
                            ))
                        else:
                            for e in self.tracker.events:
                                if e.sample_id == label.sample_id:
                                    e.retrieved_ids = retrieved_ids
                                    e.retrieved_record = result
                else:
                    result = {
                        "status": db_result["status"],
                        "error":  "Customer not found or distance too high",
                        "best_distance": db_result.get("best_distance"),
                    }
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        timing["tool_exec_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return json.dumps(result), timing
