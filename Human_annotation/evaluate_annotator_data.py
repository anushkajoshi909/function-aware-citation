#!/usr/bin/env python3
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime

TRUE_SET = {"true","1","yes","y","t"}
FALSE_SET = {"false","0","no","n","f",""}

EXPECTED_COLS = [
    "uid","paper_id","question","function","gold_supported",
    "pipeline_pred_function","pipeline_pred_supported","top1_winner_pid","retrieved_snippet",
    "ann_function","ann_supported","evidence_span","notes"
]

def to_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in TRUE_SET: return True
    if s in FALSE_SET or s == "": return False
    # tolerate common spellings
    if s == "true": return True
    if s == "false": return False
    raise ValueError(f"Cannot parse boolean from: {x!r}")

def norm_label(s: str) -> str:
    """
    Normalize function labels to a compact canonical form:
    background, uses, comparison, extension, futurework, other
    """
    t = (s or "").strip().lower()
    # remove punctuation & spaces
    for ch in ["/","-","_",".",","]:
        t = t.replace(ch, " ")
    t = " ".join(t.split())  # collapse spaces

    # map common variants
    aliases = {
        "background": "background",
        "uses": "uses",
        "use": "uses",
        "comparison": "comparison",
        "compare": "comparison",
        "compares": "comparison",
        "extension": "extension",
        "extend": "extension",
        "extends": "extension",
        "motivation future work": "futurework",
        "future work": "futurework",
        "futurework": "futurework",
        "motivation": "futurework",  # often mislabeled; treat as same bucket here
        "other": "other",
        "mention": "other",
        "other mention": "other",
    }
    return aliases.get(t, t)  # fallback to cleaned token

def aggregate(rows):
    # support confusion
    tp=fp=tn=fn=0
    total = 0

    # function accuracy
    func_total = 0
    func_correct = 0

    # paper@1 among supported by annotator
    paper_total = 0
    paper_hits = 0

    for r in rows:
        # --- support metrics (annotator vs pipeline)
        gold_sup = to_bool(r.get("ann_supported", ""))
        pred_sup = to_bool(r.get("pipeline_pred_supported", ""))  # blank -> False

        total += 1
        if gold_sup and pred_sup: tp += 1
        elif (not gold_sup) and pred_sup: fp += 1
        elif (not gold_sup) and (not pred_sup): tn += 1
        elif gold_sup and (not pred_sup): fn += 1

        # --- function accuracy (normalized)
        ann_func = norm_label(r.get("ann_function", ""))
        pipe_func = norm_label(r.get("pipeline_pred_function", ""))

        if ann_func or pipe_func:
            func_total += 1
            if ann_func and pipe_func and (ann_func == pipe_func):
                func_correct += 1

        # --- paper@1: among gold-supported rows, is top1 the gold paper?
        if gold_sup:
            paper_total += 1
            if str(r.get("top1_winner_pid","")).strip() == str(r.get("paper_id","")).strip():
                paper_hits += 1

    support_accuracy = (tp+tn)/total if total else 0.0
    support_precision = (tp/(tp+fp)) if (tp+fp) else 0.0
    support_recall = (tp/(tp+fn)) if (tp+fn) else 0.0
    support_f1 = (2*support_precision*support_recall/(support_precision+support_recall)) if (support_precision+support_recall) else 0.0
    function_accuracy = (func_correct/func_total) if func_total else 0.0
    paper_recall_at1 = (paper_hits/paper_total) if paper_total else 0.0

    return {
        "n_functions": total,
        "support_confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "support_accuracy": support_accuracy,
        "support_precision": support_precision,
        "support_recall": support_recall,
        "support_f1": support_f1,
        "function_accuracy_vs_annotator": function_accuracy,
        "paper_recall@1_vs_annotator": paper_recall_at1
    }

def main():
    ap = argparse.ArgumentParser(description="Evaluate pipeline vs. ANNOTATED CSV (ann_function/ann_supported as gold).")
    ap.add_argument("--csv", required=True, help="Annotated CSV path.")
    ap.add_argument("--out", default="", help="Optional summary JSON output path.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in EXPECTED_COLS if c not in reader.fieldnames]
        if missing:
            # allow missing retrieved_snippet/evidence/notes; warn only
            required = set(EXPECTED_COLS) - {"retrieved_snippet","evidence_span","notes"}
            missing_req = [c for c in required if c not in reader.fieldnames]
            if missing_req:
                raise SystemExit(f"CSV is missing required columns: {missing_req}\nFound: {reader.fieldnames}")
        rows = list(reader)

    metrics = aggregate(rows)

    out_path = Path(args.out) if args.out else csv_path.with_name(
        f"summary_annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "source_csv": str(csv_path)}, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\n📄 Saved summary JSON: {out_path}")

if __name__ == "__main__":
    main()
