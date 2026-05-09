#!/usr/bin/env python3
import argparse, json, csv
from pathlib import Path
from collections import OrderedDict, defaultdict

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except json.JSONDecodeError:
                continue

def load_perfunc_csv(csv_path):
    """
    Expected columns (from your earlier evaluator CSV):
      idx,paper_id,function,gt_supported,pred_supported,function_correct,paper_correct@1,hit_any,winner_pid,run_dir
    Returns a dict keyed by (paper_id, function_lower) -> row dict
    """
    m = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pid = str(r.get("paper_id","")).strip()
            func = str(r.get("function","")).strip().lower()
            if not pid or not func:
                continue
            m[(pid, func)] = r
    return m

def try_load_snippet(run_dir, func_name):
    """
    Open run_dir/function_selection_results.jsonl and extract result.answer_sentence for the matching function.
    """
    try:
        p = Path(run_dir) / "function_selection_results.jsonl"
        if not p.exists():
            return ""
        with open(p, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                f_name = str(rec.get("function","")).strip().lower()
                if f_name == func_name.lower():
                    result = rec.get("result") or {}
                    return str(result.get("answer_sentence","") or "")
    except Exception:
        pass
    return ""

def main():
    ap = argparse.ArgumentParser(description="Generate an annotation sheet (first N papers, 2 rows per paper).")
    ap.add_argument("--dataset", required=True, help="Path to dataset JSONL (with keys: paper_id, question, citation_functions, labels, supported_functions, unsupported_functions).")
    ap.add_argument("--out_csv", required=True, help="Output CSV path.")
    ap.add_argument("--n_papers", type=int, default=50)
    ap.add_argument("--perfunc_csv", help="Optional: per-function pipeline CSV to prefill pipeline_pred_* (idx,paper_id,function,gt_supported,pred_supported,...).")
    ap.add_argument("--with_snippets", action="store_true", help="If set, try to read retrieved_snippet from run_dir/function_selection_results.jsonl (requires perfunc_csv).")
    args = ap.parse_args()

    # Map for pipeline predictions
    perfunc_map = {}
    if args.perfunc_csv:
        perfunc_map = load_perfunc_csv(args.perfunc_csv)

    # Collect first N unique papers
    papers = OrderedDict()
    for rec in read_jsonl(args.dataset):
        pid = rec.get("paper_id")
        if not pid:
            continue
        if pid not in papers:
            papers[pid] = rec
        if len(papers) >= args.n_papers:
            break

    rows = []
    for pid, rec in papers.items():
        # Decide functions and gold supports
        label_pairs = []
        labels = rec.get("labels") or []
        if labels and isinstance(labels, list):
            for lab in labels[:2]:
                func = str(lab.get("function","")).strip()
                sup = bool(lab.get("supported", False))
                label_pairs.append((func, sup))
        else:
            # fallback from citation_functions + supported/unsupported lists
            cf = (rec.get("citation_functions") or [])[:2]
            supported = set(rec.get("supported_functions") or [])
            for func in cf:
                label_pairs.append((str(func).strip(), (func in supported)))

        # Ensure at most 2 rows
        label_pairs = label_pairs[:2]

        # Build rows
        for idx, (func, gold_sup) in enumerate(label_pairs, 1):
            uid = f"{pid}#{idx}"
            question = rec.get("question","")

            # Defaults
            pipeline_pred_function = ""
            pipeline_pred_supported = ""
            top1_winner_pid = ""
            retrieved_snippet = ""

            # Try to prefill from per-function CSV
            key = (pid, func.strip().lower())
            match = perfunc_map.get(key)
            if match:
                pipeline_pred_function = func  # we key by function already; leave as-is
                pipeline_pred_supported = str(match.get("pred_supported","")).strip()
                top1_winner_pid = str(match.get("winner_pid","")).strip()

                if args.with_snippets:
                    run_dir = str(match.get("run_dir","")).strip()
                    if run_dir:
                        retrieved_snippet = try_load_snippet(run_dir, func)

            rows.append({
                "uid": uid,
                "paper_id": pid,
                "question": question,
                "function": func,                # function for this row
                "gold_supported": "TRUE" if gold_sup else "FALSE",
                "pipeline_pred_function": pipeline_pred_function,
                "pipeline_pred_supported": pipeline_pred_supported,
                "top1_winner_pid": top1_winner_pid,
                "retrieved_snippet": retrieved_snippet,
                "ann_function": "",              # annotators fill
                "ann_supported": "",             # annotators fill
                "evidence_span": "",             # annotators fill when ann_supported=TRUE
                "notes": ""                      # annotators fill
            })

    # Write CSV
    import pandas as pd
    cols = [
        "uid","paper_id","question","function","gold_supported",
        "pipeline_pred_function","pipeline_pred_supported","top1_winner_pid","retrieved_snippet",
        "ann_function","ann_supported","evidence_span","notes"
    ]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} rows for {len(papers)} papers to {args.out_csv}")

if __name__ == "__main__":
    main()
