#!/usr/bin/env python3
"""
Baseline RAG (no function-awareness) with richer decision output:
- For each dataset row (paper_id, question, ...):
  1) Write a minimal classified_outputs.jsonl containing the question.
  2) Run your retrieval script to produce top-k candidates.
  3) Ask the LLM: answer + cite EXACTLY ONE paper_id from the candidates.
     Returns JSON with: answer (ends with (paper_id)), citation_paper_id, explanation, evidence_quote.
  4) Save per-run artifacts and aggregate metrics:
     - citation_accuracy@1 (predicted paper_id == gold paper_id)
     - retrieval_recall@K (gold in top-K)

Assumptions:
- Retrieval script: RetrievalAugmentedGeneration/Retreival_query_based.py
- It writes: RetrievalAugmentedGeneration/outputs/topk_candidates_query.jsonl
- FAISS + meta configured via INDEX_PATH, META_PATH (defaults set below).
"""

import os
import sys
import json
import csv
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# ---------- Defaults / Paths ----------
ROOT = Path("/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project").resolve()
RAG  = ROOT / "RetrievalAugmentedGeneration"
RETRIEVAL_PY = RAG / "Retreival_query_based.py"

CLASSIFIED_JSONL = RAG / "classified_outputs.jsonl"
TOPK_PATH_RAG    = RAG / "outputs" / "topk_candidates_query.jsonl"

RUNS_DIR_DEFAULT = ROOT / "eval_runs_baseline"
MODEL_DEFAULT    = "meta-llama/Llama-3.3-70B-Instruct"
API_KEY_FILE     = Path.home() / ".scadsai-api-key"
TOPK_DEFAULT     = 10

# ---------- Env defaults ----------
os.environ.setdefault("SKIP_REINDEX", "1")
os.environ.setdefault("INDEX_PATH", str(ROOT / "e5_index_subset_1" / "index.faiss"))
os.environ.setdefault("META_PATH",  str(ROOT / "e5_index_subset_1" / "meta.parquet"))

# ---------- LLM client ----------
try:
    from openai import OpenAI
except ImportError as e:
    raise SystemExit("pip install openai") from e

def load_client():
    if not API_KEY_FILE.exists():
        raise SystemExit(f"❌ API key file not found at {API_KEY_FILE}")
    key = API_KEY_FILE.read_text(encoding="utf-8").strip()
    return OpenAI(base_url="https://llm.scads.ai/v1", api_key=key)

CLIENT = load_client()

def llm_json(prompt_user: str, model: str, temperature: float = 0.0, max_tokens: int = 600) -> Dict[str, Any]:
    resp = CLIENT.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return STRICT JSON only. No prose, no markdown, no code fences."},
            {"role": "user", "content": prompt_user}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    raw = (resp.choices[0].message.content or "").strip()
    # tolerant parse
    try:
        return json.loads(raw)
    except Exception:
        s = raw.strip().strip("`")
        i = s.find("{")
        if i >= 0:
            depth = 0
            for j, ch in enumerate(s[i:], start=i):
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(s[i:j+1])
                        except Exception:
                            break
    return {"answer": "", "citation_paper_id": "", "explanation": "", "evidence_quote": ""}

# ---------- Utils ----------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return rows

def load_dataset_jsonl(path: Path) -> List[Dict[str, Any]]:
    return read_jsonl(path)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def run(cmd, cwd=None, env=None):
    print(f"-> {' '.join(map(str, cmd))} (cwd={cwd or os.getcwd()})")
    return subprocess.run(cmd, cwd=cwd, env=env, check=False,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

def minimal_classified_jsonl(question: str) -> Dict[str, Any]:
    # Minimal structure your retriever expects (latest record)
    return {
        "query": question,
        "citation_function_classification": {
            "citation_functions": []  # baseline: no functions
        },
        "timestamp": datetime.utcnow().isoformat()
    }

def make_prompt(question: str, candidates: List[Dict[str, Any]]) -> str:
    # Build candidate block (up to 10 shown)
    blocks = []
    for i, c in enumerate(candidates, 1):
        pid = c.get("paper_id") or c.get("arxiv_id") or c.get("id") or f"cand_{i}"
        title = (c.get("title") or "")[:800]
        abstract = (c.get("abstract_full") or c.get("abstract") or "")[:4000]
        blocks.append(f"""{i}) {pid} — {title}
Abstract: {abstract}
""")
    joined = "\n".join(blocks)
    return f"""You are given up to 10 candidate paper abstracts. 
Answer the user's question briefly and cite EXACTLY ONE paper_id from the candidates.

Return STRICT JSON ONLY with fields:
{{
  "answer": "one or two concise sentences that END with the cited paper id in parentheses, e.g. ... (hep-ph/0102325).",
  "citation_paper_id": "<exact paper_id from the candidates>",
  "explanation": "one or two sentences explaining why this paper best supports the answer",
  "evidence_quote": "≤30 words copied verbatim from the title/abstract that supports the answer"
}}

Rules:
- Cite exactly ONE paper from the candidates.
- If none is relevant, return:
  {{"answer":"insufficient evidence","citation_paper_id":"","explanation":"","evidence_quote":""}}
- The quote must be an exact substring from the candidate title/abstract (≤30 words).
- Do not include any text outside the JSON.

QUESTION:
{question}

CANDIDATES:
{joined}
"""

# ---------- Metrics ----------
def eval_batch(rows: List[Dict[str, Any]], k: int):
    total = len(rows)
    acc1 = sum(1 for r in rows if r.get("predicted_paper_id") == r.get("gold_paper_id"))
    # retrieval recall@k uses topk_list in each row
    ret_rows = [r for r in rows if r.get("topk_list")]
    hitk = sum(1 for r in ret_rows if r["gold_paper_id"] in r["topk_list"][:k])
    return {
        "n_rows": total,
        "citation_accuracy@1": (acc1 / total) if total else 0.0,
        f"retrieval_recall@{k}": (hitk / len(ret_rows)) if ret_rows else 0.0,
        "n_with_topk": len(ret_rows)
    }

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Baseline RAG (no functions): cite exactly one paper and evaluate correctness.")
    ap.add_argument("--dataset", required=True, help="Path to dataset JSONL (with fields: paper_id, question, citation_functions...).")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR_DEFAULT), help="Folder to store per-run outputs.")
    ap.add_argument("--topk", type=int, default=TOPK_DEFAULT, help="How many abstracts to show the LLM (<=10 recommended).")
    ap.add_argument("--retrieval-topk", type=int, default=20, help="How many to retrieve from FAISS (retrieval stage).")
    ap.add_argument("--model", default=MODEL_DEFAULT, help="LLM model name.")
    ap.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit.")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    runs_dir = Path(args.runs_dir)
    ensure_dir(runs_dir)

    # Load dataset
    ds = load_dataset_jsonl(dataset_path)
    if args.limit:
        ds = ds[:args.limit]

    results_rows = []

    for i, row in enumerate(ds, 1):
        paper_id = row.get("paper_id", f"UNK_{i}")
        question = row.get("question", "")
        # Duplicate per declared function (to mirror 2-rows-per-paper tallies)
        cfuncs = row.get("citation_functions") or []
        if not cfuncs:
            cfuncs = ["_baseline_"]

        subdir = runs_dir / f"{i:04d}_{paper_id.replace('/','_')}"
        ensure_dir(subdir)

        print(f"\n[{i}/{len(ds)}] ▶ Baseline RAG for paper_id={paper_id}")
        print(f"    Q: {question}")

        # 1) Write minimal classified_outputs.jsonl
        classified_record = minimal_classified_jsonl(question)
        ensure_dir(RAG / "outputs")
        CLASSIFIED_JSONL.write_text(json.dumps(classified_record) + "\n", encoding="utf-8")

        # 2) Retrieval
        env = {**os.environ}
        cmd = [
            sys.executable, str(RETRIEVAL_PY),
            "--skip-reindex",
            "--index", env["INDEX_PATH"],
            "--meta",  env["META_PATH"],
            "--topk", str(max(args.topk, args.retrieval_topk))
        ]
        ret = run(cmd, cwd=str(RAG), env=env)
        (subdir / "retrieval_stdout.txt").write_text(ret.stdout, encoding="utf-8")

        if ret.returncode != 0 or not TOPK_PATH_RAG.exists():
            print("   ⚠️ retrieval failed or no candidates; continuing with empty candidates.")
            candidates = []
        else:
            # read candidates
            candidates = read_jsonl(TOPK_PATH_RAG)
            # stash a copy for provenance
            (subdir / "topk_candidates_query.jsonl").write_text(TOPK_PATH_RAG.read_text(encoding="utf-8"), encoding="utf-8")

        # 3) Prompt with up to args.topk candidates
        cand_sorted = sorted(candidates, key=lambda c: int(c.get("rank", 10**9)))
        shown = cand_sorted[:min(args.topk, 10)]
        topk_list = [ (c.get("paper_id") or c.get("arxiv_id") or c.get("id") or "") for c in cand_sorted ]
        topk_list = [str(x) for x in topk_list if x]

        prompt = make_prompt(question, shown)
        j = llm_json(prompt, model=args.model, temperature=0.0, max_tokens=600)
        pred_pid = (j.get("citation_paper_id") or "").strip()
        ans = (j.get("answer") or "").strip()
        explanation = (j.get("explanation") or "").strip()
        evidence_quote = (j.get("evidence_quote") or "").strip()

        # 4) Persist decision (richer)
        (subdir / "baseline_decision.json").write_text(
            json.dumps({
                "answer": ans,  # includes inline (paper_id)
                "predicted_paper_id": pred_pid,
                "explanation": explanation,
                "evidence_quote": evidence_quote,
                "topk_list": topk_list[:args.topk]
            }, indent=2),
            encoding="utf-8"
        )

        # 5) Record per-function-style rows (same gold paper across functions)
        for fx_idx, _ in enumerate(cfuncs, 1):
            uid = f"{paper_id}#{fx_idx}"
            results_rows.append({
                "uid": uid,
                "gold_paper_id": paper_id,
                "predicted_paper_id": pred_pid,
                "topk_list": topk_list[:args.topk],
                "run_dir": str(subdir)
            })

    # --- Save per-row CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    perrow_csv = runs_dir / f"baseline_perrow_{ts}.csv"
    with perrow_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["uid", "gold_paper_id", "predicted_paper_id", "topk_list", "run_dir"])
        for r in results_rows:
            w.writerow([r["uid"], r["gold_paper_id"], r["predicted_paper_id"], " ".join(r["topk_list"]), r["run_dir"]])

    # --- Aggregate metrics
    K = min(args.topk, 10)
    metrics = eval_batch(results_rows, K)

    summary_json = runs_dir / f"baseline_summary_{ts}.json"
    summary_json.write_text(json.dumps({"metrics": metrics, "perrow_csv": str(perrow_csv)}, indent=2), encoding="utf-8")

    print("\n================ BASELINE SUMMARY ================")
    print(json.dumps(metrics, indent=2))
    print(f"\n📄 Saved per-row CSV: {perrow_csv}")
    print(f"📄 Saved summary JSON: {summary_json}")
    print(f"📂 Per-run outputs in: {runs_dir}")

if __name__ == "__main__":
    main()
