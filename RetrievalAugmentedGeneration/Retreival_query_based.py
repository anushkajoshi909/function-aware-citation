#!/usr/bin/env python3
# Retreival_query_based.py — minimal port with SKIP-INDEX + local e5-only embeddings
#
# - Uses your notebook model choice exactly: intfloat/e5-small-v2 (384-d)
# - Reuses existing FAISS index + meta when --skip-reindex or SKIP_REINDEX=1
# - No SCADS/OpenAI fallback for embeddings (local-only via sentence-transformers)
# - Writes:
#     outputs/topk_candidates_query.jsonl
#     outputs/topk_candidates_query.csv
#
# NOTE: install sentence-transformers once in your env:
#   pip install -U sentence-transformers

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

# ---- Your notebook choice ----
# Model to use: 'intfloat/e5-base-v2' is a good speed/quality trade-off
E5_MODEL_NAME = "intfloat/e5-small-v2"  # as in your ipynb

# --------------------------
# JSON-safe conversion
# --------------------------
def to_jsonable(obj):
    """Convert numpy/pandas objects to plain Python types for json.dumps."""
    import numpy as _np
    import pandas as _pd
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, _np.generic):
        return obj.item()
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, _pd.Timestamp):
        return obj.isoformat()
    return obj

# --------------------------
# Small IO helpers
# --------------------------
def load_last_query(classified_path: str) -> Dict[str, Any]:
    with open(classified_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return json.loads(lines[-1])

def write_topk_jsonl(rows: List[Dict[str, Any]], out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(to_jsonable(r), ensure_ascii=False) + "\n")

def write_topk_csv(rows: List[Dict[str, Any]], out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

def load_meta(meta_path: str) -> pd.DataFrame:
    ext = Path(meta_path).suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(meta_path)
    elif ext in {".csv", ".tsv"}:
        sep = "," if ext == ".csv" else "\t"
        return pd.read_csv(meta_path, sep=sep)
    else:
        raise ValueError(f"Unsupported META_PATH extension: {ext}")

def lazy_faiss():
    import faiss  # type: ignore
    return faiss

# --------------------------
# Local-only e5 embedding
# --------------------------
def embed_query_e5(text: str, model_id: str) -> np.ndarray:
    """
    Local sentence-transformers only. No remote fallback.
    - Adds e5 'query: ' prefix
    - Returns (1, d) float32 normalized (e5-small-v2 => d=384)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit(
            "❌ sentence-transformers is not installed.\n"
            "   Install it and rerun:\n"
            "   pip install -U sentence-transformers\n"
        )
    q = f"query: {text}"
    model = SentenceTransformer(model_id, device="cpu")  # set "cuda" for GPU if desired
    vec = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype("float32")

# --------------------------
# FAISS search (skip-index flow)
# --------------------------
def search_faiss(index_path: str, meta_path: str, qvec: np.ndarray, topk: int) -> List[Dict[str, Any]]:
    faiss = lazy_faiss()
    index = faiss.read_index(index_path)

    if qvec.shape[1] != index.d:
        raise SystemExit(
            f"❌ Embedding dim {qvec.shape[1]} != index dim {index.d}. "
            f"Your index must be built with the same model (e5-small-v2 = 384-d)."
        )

    D, I = index.search(qvec.astype("float32"), topk)
    meta = load_meta(meta_path)

    rows: List[Dict[str, Any]] = []
    for rank, (dist, idx) in enumerate(zip(D[0].tolist(), I[0].tolist())):
        if idx is None or idx < 0 or idx >= len(meta):
            continue
        r = meta.iloc[idx]

        # choose best abstract field
        abstract_full = r.get("abstract_full")
        abstract = r.get("abstract")
        abs_text = abstract_full if isinstance(abstract_full, str) and abstract_full.strip() else (abstract or "")

        # normalize authors for JSON
        authors = r.get("authors", "")
        if isinstance(authors, (list, tuple)):
            authors = ", ".join(map(str, authors))
        elif isinstance(authors, np.ndarray):
            authors = ", ".join(map(str, authors.tolist()))

        year = r.get("year", "")
        year = str(year) if year is not None else ""

        rows.append({
            "rank": int(rank),
            "score": float(dist),
            "paper_id": r.get("paper_id") or r.get("arxiv_id") or r.get("id"),
            "title": r.get("title", ""),
            "abstract": abstract or "",
            "abstract_full": abs_text,
            "authors": authors,
            "year": year,
            "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
    return rows

# --------------------------
# (Optional) Original rebuild path stub
# --------------------------
def reindex_and_search(query: str, topk: int) -> List[Dict[str, Any]]:
    """Disabled to prevent accidental dataset scan. Wire your notebook logic here if needed."""
    raise RuntimeError(
        "Rebuild path disabled. Run with --skip-reindex or set SKIP_REINDEX=1 "
        "to reuse your FAISS index + meta."
    )

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classified", type=str, default="classified_outputs.jsonl")
    ap.add_argument("--out-jsonl", type=str, default="outputs/topk_candidates_query.jsonl")
    ap.add_argument("--out-csv",   type=str, default="outputs/topk_candidates_query.csv")
    ap.add_argument("--topk", type=int, default=10)  # default to top-10 as requested

    # Skip & resources
    ap.add_argument("--skip-reindex", action="store_true")
    ap.add_argument("--index", type=str, default=os.getenv("INDEX_PATH", "index.faiss"))
    ap.add_argument("--meta",  type=str, default=os.getenv("META_PATH",  "meta.parquet"))

    args = ap.parse_args()

    # 1) Query from classifier
    rec = load_last_query(args.classified)
    query = rec.get("query") or ""
    cls_funcs = ", ".join((rec.get("citation_function_classification") or {}).get("citation_functions", [])) or ""
    if not query:
        print("❌ No query found in classified_outputs.jsonl", flush=True)
        sys.exit(1)

    # 2) Skip or rebuild?
    skip_env = os.getenv("SKIP_REINDEX", "").strip() == "1"
    skip = args.skip_reindex or skip_env

    if skip:
        print("⚡ SKIP_REINDEX=1 — using existing FAISS index + meta; not scanning raw JSONL shards.", flush=True)
        if not (os.path.exists(args.index) and os.path.exists(args.meta)):
            print(f"❌ Missing index/meta: {args.index} | {args.meta}", flush=True)
            sys.exit(1)

        # Embed locally with e5-small-v2 (384-d)
        qvec = embed_query_e5(query, model_id=E5_MODEL_NAME)

        # Search and write outputs
        rows = search_faiss(args.index, args.meta, qvec, topk=args.topk)
        write_topk_jsonl(rows, args.out_jsonl)
        write_topk_csv([{**r, "classifier_functions": cls_funcs} for r in rows], args.out_csv)
        print(f"✅ Top-k candidates (FAISS): {os.path.abspath(args.out_jsonl)}", flush=True)
        return

    # Rebuild (disabled)
    rows = reindex_and_search(query=query, topk=args.topk)
    write_topk_jsonl(rows, args.out_jsonl)
    write_topk_csv([{**r, "classifier_functions": cls_funcs} for r in rows], args.out_csv)

if __name__ == "__main__":
    main()
