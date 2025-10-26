#!/usr/bin/env python3
import os
import sys
import json
import csv
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# --- Defaults (tweak if your project lives elsewhere)
PROJECT_ROOT = Path(__file__).resolve().parent
PIPELINE_PATH_DEFAULT = PROJECT_ROOT / "run_pipeline.py"
RUNS_DIR_DEFAULT = PROJECT_ROOT / "eval_runs_3_deepseek"  # where we stash per-example outputs

# --- Filenames your pipeline already produces
FUNC_RESULTS_JSONL = "RetrievalAugmentedGeneration/outputs/function_selection_results.jsonl"
FUNC_ANSWERS_TXT   = "RetrievalAugmentedGeneration/outputs/function_answer_unified.txt"
CLASSIFIED_JSONL   = "RetrievalAugmentedGeneration/classified_outputs.jsonl"  # used by pipeline

# --- Ensure retrieval reuses existing index/meta by default ---
os.environ.setdefault("SKIP_REINDEX", "1")
# Paths to existing index + meta (adjust if yours live elsewhere)
os.environ.setdefault("INDEX_PATH", str("/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/e5_index_subset_1/index.faiss"))
os.environ.setdefault("META_PATH",  str("/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/e5_index_subset_1/meta.parquet"))

# --------------------------
# Helpers
# --------------------------
def read_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return rows

def read_last_jsonl(path: Path):
    """Read just the last valid JSON line."""
    last = None
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                last = json.loads(ln)
            except json.JSONDecodeError:
                continue
    return last

def load_dataset(path: Path):
    """Your dataset lines already contain:
       paper_id, question, citation_functions, answer, explanation,
       labels (list of {"function":..., "supported": bool})
    """
    return read_jsonl(path)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        return True
    return False

def is_negative_answer_sentence(answer_sentence: str, func_name: str):
    """Safety check if you ever want to infer 'unsupported' from text.
       We don't rely on it for evaluation, but keep for debugging."""
    neg_markers = [
        "does not provide information relevant",
        "doesn't provide information relevant",
        "no information relevant",
        "not provide information relevant",
    ]
    s = (answer_sentence or "").lower()
    if any(x in s for x in neg_markers) and func_name.lower() in s:
        return True
    return False

# --------------------------
# Evaluation
# --------------------------
def get_predicted_functions(classified_jsonl_path: Path):
    """
    Pull the classifier's predicted citation functions from classified_outputs.jsonl.
    We take the LAST record (the one from the most recent run).
    """
    if not classified_jsonl_path.exists():
        return []
    rec = read_last_jsonl(classified_jsonl_path)
    if not rec:
        return []
    clf = rec.get("citation_function_classification") or {}
    funcs = clf.get("citation_functions") or []
    # normalize capitalization/whitespace just in case
    return [str(f).strip() for f in funcs if f]

def evaluate_one_example(gt_row: dict, results_jsonl_path: Path, classified_jsonl_path: Path):
    """
    gt_row schema (from your dataset):
      - paper_id (gold)
      - question
      - citation_functions (list of 2)
      - labels: [{"function": "Uses", "supported": true}, ...]

    results_jsonl_path schema (from your pipeline):
      - list with one or more items, each like:
        {"query": ..., "function": "Background", "result": { "function":..., "winner": {...}|null, "answer_sentence": ..., "verdicts":[...]}}
    """
    out = {
        "paper_id": gt_row.get("paper_id"),
        "question": gt_row.get("question"),
        "per_function": [],  # filled below
        "predicted_functions": []  # for debugging/visibility
    }

    # Predicted function list from classifier (independent of support)
    pred_funcs = get_predicted_functions(classified_jsonl_path)
    out["predicted_functions"] = pred_funcs[:]

    if not results_jsonl_path.exists():
        out["error"] = "missing_function_results"
        return out

    preds = read_jsonl(results_jsonl_path)
    # Map predicted results by function
    pred_by_func = {}
    for item in preds:
        func = (item.get("function") or "").strip()
        pred_by_func[func] = item.get("result") or {}

    # Build GT labels map
    gt_labels = { (lab.get("function") or "").strip(): bool(lab.get("supported", False))
                  for lab in gt_row.get("labels", []) }

    gold_pid = gt_row.get("paper_id")

    # Evaluate each function present in GT (usually 2)
    for func in gt_row.get("citation_functions", []):
        gt_supported = gt_labels.get(func, False)
        pred_result  = pred_by_func.get(func, {})
        winner = pred_result.get("winner")
        verdicts = pred_result.get("verdicts", [])

        # support predicted by pipeline selection step
        pred_supported = winner is not None  # None winner => unsupported

        # Function correctness = did classifier predict this function? (membership check)
        function_correct = (func in pred_funcs)

        # paper correctness only meaningful when GT says there *is* support and model predicted support
        paper_correct = bool(gt_supported and pred_supported and winner and (winner.get("paper_id") == gold_pid))

        # retrieval "hit-any" — did we see the correct paper in any checked candidate?
        hit_any = any(v.get("paper_id") == gold_pid for v in verdicts)

        out["per_function"].append({
            "function": func,
            "gt_supported": gt_supported,
            "pred_supported": pred_supported,
            "function_correct": function_correct,
            "paper_correct@1": paper_correct,
            "hit_any": hit_any,
            "winner_pid": (winner or {}).get("paper_id"),
        })

    return out

def aggregate_metrics(per_example_rows):
    # confusion for "supported" (over functions)
    tp = fp = tn = fn = 0
    total_funcs = 0
    paper_hits = 0
    paper_total = 0
    hit_any_cnt = 0
    hit_any_total = 0

    # optional: function membership metrics
    func_mem_total = 0
    func_mem_correct = 0

    for row in per_example_rows:
        if "per_function" not in row:
            continue
        for pf in row["per_function"]:
            total_funcs += 1
            gt = pf["gt_supported"]
            pr = pf["pred_supported"]
            if gt and pr:
                tp += 1
            elif (not gt) and pr:
                fp += 1
            elif (not gt) and (not pr):
                tn += 1
            elif gt and (not pr):
                fn += 1

            # paper@1 only when both GT and pred say supported
            if gt:
                paper_total += 1
                if pf["paper_correct@1"]:
                    paper_hits += 1

            # hit-any: only meaningful if GT says supported (i.e., there exists a correct paper)
            if gt:
                hit_any_total += 1
                if pf["hit_any"]:
                    hit_any_cnt += 1

            # function membership accuracy (independent of support)
            func_mem_total += 1
            if pf["function_correct"]:
                func_mem_correct += 1

    # support classification metrics
    acc = (tp + tn) / total_funcs if total_funcs else 0.0
    prec = (tp / (tp + fp)) if (tp + fp) else 0.0
    rec = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (2*prec*rec / (prec + rec)) if (prec + rec) else 0.0

    # paper metrics
    paper_recall = (paper_hits / paper_total) if paper_total else 0.0
    hit_any = (hit_any_cnt / hit_any_total) if hit_any_total else 0.0

    # function membership accuracy
    func_membership_acc = (func_mem_correct / func_mem_total) if func_mem_total else 0.0

    return {
        "n_functions": total_funcs,
        "support_confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "support_accuracy": acc,
        "support_precision": prec,
        "support_recall": rec,
        "support_f1": f1,
        "paper_recall@1": paper_recall,
        "retrieval_hit_any": hit_any,
        "function_membership_accuracy": func_membership_acc,
    }

# --------------------------
# Batch Runner
# --------------------------
def run_pipeline_once(pipeline_path: Path, model_index: int, question: str, cwd: Path):
    """
    Feeds stdin to your interactive pipeline:
      line 1: model index
      line 2: question
    """
    env = {**os.environ}  # carry SKIP_REINDEX/INDEX_PATH/META_PATH through
    p = subprocess.run(
        [sys.executable, str(pipeline_path)],
        input=f"{model_index}\n{question}\n",
        text=True,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return p.returncode, p.stdout

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Batch-run your interactive pipeline over a dataset and evaluate.")
    ap.add_argument("--dataset", required=True, help="Path to your dataset JSONL (with labels).")
    ap.add_argument("--pipeline", default=str(PIPELINE_PATH_DEFAULT), help="Path to run_pipeline.py")
    ap.add_argument("--project-root", default=str(PROJECT_ROOT), help="Working dir where pipeline writes outputs.")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR_DEFAULT), help="Folder to collect per-example outputs.")
    ap.add_argument("--model-index", type=int, required=True, help="Menu number to select in the pipeline’s model list.")
    ap.add_argument("--limit", type=int, default=0, help="Optional: limit number of examples for a smoke test.")
    ap.add_argument("--skip-run", action="store_true", help="Only evaluate existing run folders; don't execute pipeline.")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    pipeline_path = Path(args.pipeline)
    project_root = Path(args.project_root)
    runs_dir = Path(args.runs_dir)

    ensure_dir(runs_dir)

    dataset = load_dataset(dataset_path)
    if args.limit:
        dataset = dataset[:args.limit]

    per_example_eval = []

    for i, row in enumerate(dataset, 1):
        q = row.get("question", "")
        gold_pid = row.get("paper_id", f"UNK_{i}")
        subdir = runs_dir / f"{i:04d}_{gold_pid.replace('/','_')}"
        ensure_dir(subdir)

        print(f"\n[{i}/{len(dataset)}] ▶ Running pipeline for paper_id={gold_pid}")
        print(f"    Q: {q}")

        if not args.skip_run:
            # 1) run pipeline interactively
            rc, stdout = run_pipeline_once(
                pipeline_path=pipeline_path,
                model_index=args.model_index,
                question=q,
                cwd=project_root
            )
            # save raw log
            (subdir / "pipeline_stdout.txt").write_text(stdout, encoding="utf-8")
            if rc != 0:
                print(f"   ⚠️ pipeline exited with code {rc} — continuing.")

            # 2) copy outputs away before next run overwrites them
            copied1 = copy_if_exists(project_root / FUNC_RESULTS_JSONL, subdir / "function_selection_results.jsonl")
            copied2 = copy_if_exists(project_root / FUNC_ANSWERS_TXT,   subdir / "function_answer_unified.txt")
            copied3 = copy_if_exists(project_root / CLASSIFIED_JSONL,   subdir / "classified_outputs.jsonl")
            if not (copied1 and copied2 and copied3):
                print("   ⚠️ Missing expected outputs; check pipeline logs (need function_results, answers, and classified).")

        # 3) evaluate this example using the copied results
        one_eval = evaluate_one_example(
            gt_row=row,
            results_jsonl_path=subdir / "function_selection_results.jsonl",
            classified_jsonl_path=subdir / "classified_outputs.jsonl",
        )
        one_eval["run_dir"] = str(subdir)
        per_example_eval.append(one_eval)

    # --- aggregate & save
    metrics = aggregate_metrics(per_example_eval)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_json = runs_dir / f"summary_{ts}.json"
    perrow_csv    = runs_dir / f"per_function_{ts}.csv"

    # save summary
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "rows": per_example_eval}, f, indent=2)

    # save per-function CSV
    with open(perrow_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "idx","paper_id","function",
            "gt_supported","pred_supported",
            "function_correct",  # membership in classifier's predicted functions
            "paper_correct@1","hit_any","winner_pid","run_dir"
        ])
        idx = 0
        for i, row in enumerate(per_example_eval, 1):
            pid = row.get("paper_id")
            rdir = row.get("run_dir", "")
            for pf in row.get("per_function", []):
                idx += 1
                w.writerow([
                    idx, pid, pf["function"], pf["gt_supported"], pf["pred_supported"],
                    pf["function_correct"], pf["paper_correct@1"], pf["hit_any"], pf.get("winner_pid",""), rdir
                ])

    print("\n================ EVAL SUMMARY ================")
    print(json.dumps(metrics, indent=2))
    print(f"\n📄 Saved summary JSON: {summary_json}")
    print(f"📄 Saved per-function CSV: {perrow_csv}")
    print(f"📂 Per-run outputs in: {runs_dir}")

if __name__ == "__main__":
    main()
