#!/usr/bin/env python3
import os, sys, subprocess, shutil, time
from pathlib import Path

# ---------- Environment ----------
# Force reuse of existing FAISS + meta (no shard scanning)
os.environ.setdefault("SKIP_REINDEX", "1")

# ---------- Paths ----------
ROOT   = Path("/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project").resolve()
RAG    = ROOT / "RetrievalAugmentedGeneration"

CLASSIFIER   = RAG / "classifying_question.py"
RETRIEVAL_PY = RAG / "Retreival_query_based.py"   # direct .py (no nbconvert)
FINAL_STAGE  = RAG / "function_based_answer-2.py" # relaxed-only, top-K by rank

CLASSIFIED   = RAG / "classified_outputs.jsonl"
TOPK_RAG     = RAG / "outputs" / "topk_candidates_query.jsonl"
TOPK_ROOT    = ROOT / "outputs" / "topk_candidates_query.jsonl"  # optional compatibility link

# Paths to existing index + meta (adjust if yours live elsewhere)
os.environ.setdefault("INDEX_PATH", str("/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/e5_index_subset_1/index.faiss"))
os.environ.setdefault("META_PATH",  str("/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/e5_index_subset_1/meta.parquet"))

# ---------- Helpers ----------
def run(cmd, cwd=None, env=None):
    print(f"-> {' '.join(map(str, cmd))}  (cwd={cwd or os.getcwd()})")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)

def safe_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            dst.unlink()
        except FileNotFoundError:
            pass
    try:
        dst.symlink_to(src)
        print(f"🔗 Symlinked: {dst} -> {src}")
    except Exception as e:
        print(f"Symlink failed ({e}); copying instead.")
        shutil.copyfile(src, dst)
        print(f"📄 Copied: {src} -> {dst}")

# ---------- Pipeline ----------
def main():
    # Ensure output dirs exist
    (RAG / "outputs").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)

    env = {**os.environ}

    # 1) Classify (interactive/whatever your script does)
    run([sys.executable, str(CLASSIFIER)], cwd=str(RAG), env=env)
    if not CLASSIFIED.exists():
        raise FileNotFoundError(f"Expected {CLASSIFIED}")
    print("✅ Saved to classified_outputs.jsonl")
    print("🔧 Ready for retrieval module (.py).")

    # 2) Retrieval (.py) — reuse FAISS + meta
    retrieval_cmd = [
        sys.executable, str(RETRIEVAL_PY),
        "--skip-reindex",
        "--index", env["INDEX_PATH"],
        "--meta",  env["META_PATH"],
        "--topk", "20"
    ]
    run(retrieval_cmd, cwd=str(RAG), env=env)

    # 3) Verify retrieval outputs & create canonical link
    if not TOPK_RAG.exists():
        raise FileNotFoundError(f"Expected retrieval output at {TOPK_RAG}")
    print(f"✅ Top-k candidates: {TOPK_RAG}")
    safe_symlink(TOPK_RAG, TOPK_ROOT)

    # 4) Final stage (relaxed-only + top-K by rank)
    time.sleep(1)
    try:
        run([sys.executable, str(FINAL_STAGE), "--debug", "--max-check", "10"], cwd=str(RAG), env=env)
    except subprocess.CalledProcessError:
        print("⚠️ Final stage failed once; pausing and retrying…")
        time.sleep(3)
        run([sys.executable, str(FINAL_STAGE), "--debug", "--max-check", "10"], cwd=str(RAG), env=env)

    print("\n🎉 Pipeline complete.")
    print(f"  • Classified file : {CLASSIFIED}")
    print(f"  • TopK (canonical): {TOPK_RAG}")
    print(f"  • Final outputs   : {RAG / 'outputs'}")

if __name__ == "__main__":
    main()
