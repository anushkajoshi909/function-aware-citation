#!/usr/bin/env python3
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime

EXPECTED_COLS = [
    "idx","paper_id","function",
    "gt_supported","pred_supported",
    "function_correct",
    "paper_correct@1","hit_any",
    "winner_pid","run_dir"
]

TRUE_SET = {"true","1","yes","y","t"}
FALSE_SET = {"false","0","no","n","f",""}

def to_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in TRUE_SET:
        return True
    if s in FALSE_SET:
        return False
    # fallback: try python-style
    if s == "none":
        return False
    if s == "true":
        return True
    if s == "false":
        return False
    raise ValueError(f"Cannot parse boolean from: {x!r}")

def aggregate_metrics(rows):
    # Same logic as your integrated aggregator
    tp = fp = tn = fn = 0
    total_funcs = 0

    paper_hits = 0
    paper_total = 0

    hit_any_cnt = 0
    hit_any_total = 0

    func_mem_total = 0
    func_mem_correct = 0

    for r in rows:
        total_funcs += 1
        gt = to_bool(r["gt_supported"])
        pr = to_bool(r["pred_supported"])
        func_correct = to_bool(r["function_correct"])
        paper_at1 = to_bool(r["paper_correct@1"])
        hit_any = to_bool(r["hit_any"])

        if gt and pr:
            tp += 1
        elif (not gt) and pr:
            fp += 1
        elif (not gt) and (not pr):
            tn += 1
        elif gt and (not pr):
            fn += 1

        if gt:
            paper_total += 1
            if paper_at1:
                paper_hits += 1

        if gt:
            hit_any_total += 1
            if hit_any:
                hit_any_cnt += 1

        func_mem_total += 1
        if func_correct:
            func_mem_correct += 1

    acc = (tp + tn) / total_funcs if total_funcs else 0.0
    prec = (tp / (tp + fp)) if (tp + fp) else 0.0
    rec = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    paper_recall_at1 = (paper_hits / paper_total) if paper_total else 0.0
    retrieval_hit_any = (hit_any_cnt / hit_any_total) if hit_any_total else 0.0
    func_membership_acc = (func_mem_correct / func_mem_total) if func_mem_total else 0.0

    return {
        "n_functions": total_funcs,
        "support_confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "support_accuracy": acc,
        "support_precision": prec,
        "support_recall": rec,
        "support_f1": f1,
        "paper_recall@1": paper_recall_at1,
        "retrieval_hit_any": retrieval_hit_any,
        "function_membership_accuracy": func_membership_acc,
    }

def main():
    ap = argparse.ArgumentParser(description="Compute evaluation metrics from a per-function CSV (no pipeline run).")
    ap.add_argument("--csv", required=True, help="Path to per-function CSV.")
    ap.add_argument("--out", default="", help="Optional path to write a summary JSON. Defaults next to the CSV.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # sanity check headers
        missing = [c for c in EXPECTED_COLS if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f"CSV is missing required columns: {missing}\nFound: {reader.fieldnames}")
        rows = list(reader)

    metrics = aggregate_metrics(rows)

    # default summary path
    if args.out:
        out_path = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = csv_path.with_name(f"summary_from_csv_{ts}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "source_csv": str(csv_path)}, f, indent=2)

    # Pretty-print to console too
    print(json.dumps(metrics, indent=2))
    print(f"\n📄 Saved summary JSON: {out_path}")

if __name__ == "__main__":
    main()
