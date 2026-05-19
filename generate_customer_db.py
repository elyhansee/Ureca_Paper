#!/usr/bin/env python3
"""
generate_customer_db.py
=======================
Generates a synthetic telecom customer database of N records (default 1M)
and saves it as a Parquet file for fast loading, plus builds and persists
a FAISS IVF index over phone-number embeddings.

Usage:
    python generate_customer_db.py --n 1_000_000 --out ./customer_db
    python generate_customer_db.py --n 10_000_000 --out ./customer_db

Outputs:
    ./customer_db/customers.parquet   – columnar customer records
    ./customer_db/faiss.index         – trained FAISS IVFFlat index
    ./customer_db/id_map.npy          – row-index → customer_id mapping
    ./customer_db/meta.json           – generation metadata
"""

import os
import json
import time
import argparse
import random
import string
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM     = 384
BATCH_SIZE    = 4096          # embedding batch size
IVF_NLIST     = 4096          # FAISS IVF cluster count (sqrt(N) rule of thumb)
IVF_NPROBE    = 64            # search-time clusters to inspect
SEED          = 42

PLANS         = ["Basic", "Standard", "Premium", "Enterprise", "Family"]
PLAN_WEIGHTS  = [0.35,    0.30,       0.20,      0.10,         0.05]

COUNTRY_CODES = ["+1", "+44", "+49", "+33", "+61", "+81", "+55", "+91"]
CC_WEIGHTS    = [0.50, 0.10,  0.08,  0.07,  0.05,  0.05,  0.05,  0.10]

FIRST_NAMES = [
    "James","Mary","John","Patricia","Robert","Jennifer","Michael","Linda",
    "William","Barbara","David","Elizabeth","Richard","Susan","Joseph","Jessica",
    "Thomas","Sarah","Charles","Karen","Christopher","Lisa","Daniel","Nancy",
    "Matthew","Betty","Anthony","Margaret","Mark","Sandra","Donald","Ashley",
    "Steven","Emily","Paul","Kimberly","Andrew","Donna","Kenneth","Carol",
    "George","Michelle","Joshua","Amanda","Kevin","Dorothy","Brian","Melissa",
    "Edward","Deborah","Ronald","Stephanie","Timothy","Rebecca","Jason","Sharon",
    "Jeffrey","Laura","Ryan","Cynthia","Jacob","Kathleen","Gary","Amy",
    "Nicholas","Angela","Eric","Shirley","Jonathan","Anna","Stephen","Brenda",
    "Larry","Pamela","Justin","Emma","Scott","Nicole","Brandon","Helen",
    "Benjamin","Samantha","Samuel","Katherine","Frank","Christine","Gregory",
    "Debra","Raymond","Rachel","Alexander","Carolyn","Patrick","Janet",
    "Jack","Catherine","Dennis","Maria","Jerry","Heather","Tyler","Diane",
]

LAST_NAMES = [
    "Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
    "Rodriguez","Martinez","Hernandez","Lopez","Gonzalez","Wilson","Anderson",
    "Thomas","Taylor","Moore","Jackson","Martin","Lee","Perez","Thompson",
    "White","Harris","Sanchez","Clark","Ramirez","Lewis","Robinson","Walker",
    "Young","Allen","King","Wright","Scott","Torres","Nguyen","Hill","Flores",
    "Green","Adams","Nelson","Baker","Hall","Rivera","Campbell","Mitchell",
    "Carter","Roberts","Turner","Phillips","Evans","Collins","Edwards","Stewart",
    "Morris","Murphy","Cook","Rogers","Morgan","Peterson","Cooper","Reed",
    "Bailey","Bell","Gomez","Kelly","Howard","Ward","Cox","Diaz","Richardson",
    "Wood","Watson","Brooks","Bennett","Gray","James","Reyes","Cruz","Hughes",
    "Price","Myers","Long","Foster","Sanders","Ross","Morales","Powell",
    "Sullivan","Russell","Ortiz","Jenkins","Gutierrez","Perry","Butler","Barnes",
]


# ── Embedder ──────────────────────────────────────────────────────────────────
class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str = "cpu"):
        self.device = device
        print(f"  Loading embedder: {model_name} → {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=32, return_tensors="pt"
        ).to(self.device)
        out = self.model(**enc)
        tok = out[0]
        mask = enc["attention_mask"].unsqueeze(-1).expand(tok.size()).float()
        emb = torch.sum(tok * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        emb = F.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().astype(np.float32)


# ── Data generation ───────────────────────────────────────────────────────────
def rand_phone(rng: random.Random, cc: str) -> str:
    digits = "".join(rng.choices(string.digits, k=10))
    sep = rng.choice(["", "-", " ", "."])
    if rng.random() < 0.5:
        return f"{cc}{sep}{digits[:3]}{sep}{digits[3:6]}{sep}{digits[6:]}"
    return f"{cc}{digits}"


def generate_records(n: int, seed: int = SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    print(f"  Generating {n:,} customer records...")
    t0 = time.time()

    ids          = [f"C{i:010d}" for i in range(n)]
    first_names  = rng.choices(FIRST_NAMES, k=n)
    last_names   = rng.choices(LAST_NAMES,  k=n)
    names        = [f"{f} {l}" for f, l in zip(first_names, last_names)]
    plans        = rng.choices(PLANS, weights=PLAN_WEIGHTS, k=n)
    ccs          = rng.choices(COUNTRY_CODES, weights=CC_WEIGHTS, k=n)
    phones       = [rand_phone(rng, cc) for cc in ccs]
    balances     = np.round(np_rng.uniform(-200, 500, size=n), 2)
    balance_strs = [f"${b:.2f}" if b >= 0 else f"-${abs(b):.2f}" for b in balances]

    df = pd.DataFrame({
        "id":      ids,
        "name":    names,
        "phone":   phones,
        "plan":    plans,
        "balance": balance_strs,
    })
    print(f"  Records generated in {time.time()-t0:.1f}s")
    return df


# ── FAISS index ───────────────────────────────────────────────────────────────
def build_index(df: pd.DataFrame, embedder: Embedder, out_dir: Path):
    n = len(df)
    texts = (df["name"] + " " + df["phone"]).tolist()

    print(f"  Embedding {n:,} records in batches of {BATCH_SIZE}...")
    t0 = time.time()
    all_embs = []
    for start in range(0, n, BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        embs  = embedder.encode(batch)
        all_embs.append(embs)
        if (start // BATCH_SIZE) % 50 == 0:
            pct = 100.0 * start / n
            elapsed = time.time() - t0
            eta = elapsed / max(start, 1) * (n - start)
            print(f"    {start:>10,} / {n:,}  ({pct:.1f}%)  ETA {eta:.0f}s")

    embeddings = np.vstack(all_embs).astype(np.float32)
    print(f"  Embedding done in {time.time()-t0:.1f}s  shape={embeddings.shape}")

    # Train IVFFlat (exact within cluster, approximate globally)
    print(f"  Building FAISS IVFFlat (nlist={IVF_NLIST})...")
    t1 = time.time()
    quantizer = faiss.IndexFlatL2(EMBED_DIM)
    index     = faiss.IndexIVFFlat(quantizer, EMBED_DIM, IVF_NLIST, faiss.METRIC_L2)
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = IVF_NPROBE
    print(f"  FAISS index built in {time.time()-t1:.1f}s  ntotal={index.ntotal:,}")

    faiss.write_index(index, str(out_dir / "faiss.index"))
    np.save(str(out_dir / "id_map.npy"), np.array(df["id"].tolist()))
    print(f"  Index saved → {out_dir/'faiss.index'}")
    return embeddings.shape[1]


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",   type=int, default=1_000_000,
                        help="Number of customer records to generate")
    parser.add_argument("--out", type=str, default="./customer_db",
                        help="Output directory")
    parser.add_argument("--device", default="cpu",
                        help="Embedding device (cpu / cuda)")
    parser.add_argument("--skip-faiss", action="store_true",
                        help="Skip FAISS index build (parquet only)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    df = generate_records(args.n)
    parquet_path = out_dir / "customers.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"  Saved parquet → {parquet_path}  ({parquet_path.stat().st_size/1e6:.1f} MB)")

    dim = EMBED_DIM
    if not args.skip_faiss:
        embedder = Embedder(device=args.device)
        dim = build_index(df, embedder, out_dir)

    meta = {
        "n_records":   args.n,
        "embed_model": EMBED_MODEL,
        "embed_dim":   dim,
        "ivf_nlist":   IVF_NLIST,
        "ivf_nprobe":  IVF_NPROBE,
        "seed":        SEED,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_seconds": round(time.time() - t_total, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n  Done in {meta['total_seconds']:.1f}s → {out_dir}")


if __name__ == "__main__":
    main()
