import os, json, re
import pandas as pd
from collections import defaultdict
from datetime import datetime
from openai import OpenAI

TOPK_PATH = "/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/outputs/topk_candidates.jsonl"
OUT_JSONL = "/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/outputs/llm_judgments.jsonl"
OUT_CSV   = "/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/outputs/llm_judgments.csv"
FINAL_JSONL = "/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/outputs/selected_citations.jsonl"
FINAL_CSV   = "/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/outputs/selected_citations.csv"
PER_SENTENCE_LIMIT = 10  # how many ranked candidates to show the LLM per sentence


# SCADS-style OpenAI client
api_key_path = os.path.join(os.path.expanduser("~"), ".scadsai-api-key")
if not os.path.exists(api_key_path):
    raise SystemExit("❌ API key file not found at ~/.scadsai-api-key")
with open(api_key_path) as f:
    api_key = f.read().strip()
client = OpenAI(base_url="https://llm.scads.ai/v1", api_key=api_key)

# Pick a model (same strategy you used elsewhere)
models = client.models.list().data
PREFERRED_MODEL = next((m.id for m in models if "llama" in m.id.lower()), models[0].id)
print(f"✅ LLM for judgments: {PREFERRED_MODEL}")

# ---------------------------
# Prompt + builder
# ---------------------------
RUBRIC = """
Possible citation functions:
- Background: Introduces known facts, definitions, or prior knowledge to set context.
- Uses: Describes how a known method or tool is applied in the current work.
- Compares: Compares two or more methods, tools, datasets, or results.
- Motivation: Highlights a gap, problem, or challenge that motivates the research.
- Extends: Builds upon or improves an existing method, approach, or result.
- FutureWork: Outlines work that is planned but not yet done.

Task: For the SENTENCE and requested FUNCTION, choose which candidate papers best fit the FUNCTION.

Judging guidance:
- Use only each candidate's TITLE and ABSTRACT with provided metadata (cosine, year, matches). Do not invent content.
- For Background: does the paper plausibly provide prior context/foundational knowledge?
- Consider overlap with the sentence's key terms (title_matches/abs_matches); if ANY candidate has ≥1 matched term, you MUST select at least 1 such candidate.
- Consider COSINE as a relevance hint (higher is better) and YEAR plausibility.
- Return up to 3 IDs. If NO candidate has any topical overlap and all seem irrelevant, you MAY return an empty list.

Output JSON only:
{
  "function": "<Background|Uses|Compares|Motivation|Extends|FutureWork>",
  "candidates": [
    {
      "id": <int>,
      "fit": "Fit" | "NoFit",
      "scores": { "topicality": 0-1, "function_fit": 0-1, "confidence": 0-1 },
      "rationale": "<1-2 sentences>",
      "support_snippet": "<quoted phrase from title/abstract>"
    }
  ],
  "top_k": [<up to 3 ids in ranked order>]
}
"""


def focus_snippet(text, terms, window=80, max_len=700, fallback_len=600):
    """
    Build a compact snippet from `text` around occurrences of `terms`.
    - window: chars kept on each side of each hit
    - max_len: hard cap for the final snippet
    - fallback_len: if no hits, take leading portion of text
    """
    if not text:
        return ""
    t = str(text)
    # Normalize/clean terms; keep unique lowercase tokens ≥3 chars
    uniq_terms = []
    seen = set()
    for term in (terms or []):
        k = term.strip().lower()
        if len(k) >= 3 and k not in seen:
            seen.add(k)
            uniq_terms.append(k)
    # Gather hit windows
    spans = []
    for term in uniq_terms:
        for m in re.finditer(re.escape(term), t, flags=re.I):
            start = max(0, m.start() - window)
            end   = min(len(t), m.end() + window)
            spans.append([start, end])
    # If no hits, return a trimmed head
    if not spans:
        head = t[:fallback_len]
        return head if len(t) <= fallback_len else head.rstrip() + "…"
    # Merge overlapping spans
    spans.sort()
    merged = [spans[0]]
    for s, e in spans[1:]:
        last = merged[-1]
        if s <= last[1] + 10:  # small glue
            last[1] = max(last[1], e)
        else:
            merged.append([s, e])
    # Assemble with ellipses, capped by max_len
    parts = []
    total = 0
    for s, e in merged:
        seg = t[s:e].strip()
        if not seg:
            continue
        # Add ellipses if not at boundaries
        if s > 0: seg = "…" + seg
        if e < len(t): seg = seg + "…"
        if total + len(seg) > max_len:
            seg = seg[: max(0, max_len - total)].rstrip() + "…"
            parts.append(seg)
            break
        parts.append(seg)
        total += len(seg)
        if total >= max_len:
            break
    out = " ".join(parts).strip()
    return out or (t[:fallback_len].rstrip() + ("…" if len(t) > fallback_len else ""))


def build_prompt(sentence_text, func, candidates, query_terms=None, must_pick=False):
    import json
    qt = ", ".join(sorted(set((query_terms or [])))) if query_terms else ""
    lines = [RUBRIC.strip()]
    lines.append("\nSENTENCE:\n" + sentence_text.strip())
    lines.append("\nREQUESTED FUNCTION:\n" + func)
    if qt:
        lines.append("\nSENTENCE KEY TERMS (hints):\n" + qt)
    if must_pick:
        lines.append("\nADDITIONAL RULE:\nAt least one candidate shares ≥1 matched term; you MUST select at least 1 such candidate.")

    payload = []
    for i, c in enumerate(candidates):
        # Collect focus terms: title+abstract matches (+ query terms as soft hints)
        terms = []
        terms.extend(c.get("title_matches", []) or [])
        terms.extend(c.get("abs_matches", []) or [])
        # Optionally bias with sentence/query terms:
        if query_terms:
            terms.extend(query_terms)
        # Build focus snippet from abstract
        abstract_text = c.get("abstract", "") or ""
        abstract_focus = focus_snippet(
            abstract_text,
            terms=terms,
            window=80,       # tweak: 60-120 is typical
            max_len=700,     # tweak per token budget
            fallback_len=600
        )
        payload.append({
            "id": i,
            "cosine": c.get("cosine", ""),
            "year": c.get("year", ""),
            "title": c.get("title", ""),
            "abstract": abstract_focus,               # <— focused snippet here
            "title_matches": c.get("title_matches", []),
            "abs_matches": c.get("abs_matches", [])
        })

    lines.append("\nCANDIDATES:")
    lines.append(json.dumps(payload, ensure_ascii=False))
    lines.append("\nReturn only the JSON object.")
    return "\n".join(lines)

# ---------------------------
# IO helpers
# ---------------------------
def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"⚠️ Skipping malformed line {ln}")
    return rows

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def extract_json_only(text: str):
    m = re.search(r"\{.*\}\s*$", text.strip(), re.S)
    if not m:
        raise ValueError("No JSON object found in model output")
    return json.loads(m.group(0))

# ---------------------------
# LLM judging + fallback
# ---------------------------
def has_overlap(r):
    tm = set(r.get("title_matches", []) or [])
    am = set(r.get("abs_matches", []) or [])
    return bool(tm or am)

def judge_candidates(topk_rows, per_sentence_limit=PER_SENTENCE_LIMIT):
    groups = defaultdict(list)
    for r in topk_rows:
        sid = int(r.get("sentence_idx", -1))
        groups[sid].append(r)

    out_rows = []
    for sid, rows in sorted(groups.items()):
        # ranked only
        ranked = [r for r in rows if r.get("rank") is not None]
        ranked = sorted(ranked, key=lambda x: int(x.get("rank", 1_000_000)))[:per_sentence_limit]
        if not ranked:
            out_rows.append({
                "sentence_idx": sid,
                "function": rows[0].get("citation_function", ""),
                "sentence_text": rows[0].get("sentence_text", ""),
                "judgment": {"function": rows[0].get("citation_function",""), "candidates": [], "top_k": []},
                "num_candidates": 0,
                "created_at": datetime.utcnow().isoformat()
            })
            continue

        sent_text = rows[0].get("sentence_text", "")
        func      = rows[0].get("citation_function") or rows[0].get("function_requested") or "Background"
        qt        = ranked[0].get("query_terms_used", [])
        must_pick = any(has_overlap(r) for r in ranked)

        cand_views = [{
            "title": r.get("title",""),
            "abstract": r.get("abstract",""),
            "year": r.get("year",""),
            "cosine": r.get("cosine", 0.0),
            "title_matches": r.get("title_matches", []),
            "abs_matches": r.get("abs_matches", [])
        } for r in ranked]

        prompt = build_prompt(sent_text, func, cand_views, query_terms=qt, must_pick=must_pick)
        try:
            resp = client.chat.completions.create(
                model=PREFERRED_MODEL,
                messages=[{"role":"user","content": prompt}],
                temperature=0.0,
                max_tokens=700
            )
            parsed = extract_json_only(resp.choices[0].message.content)
        except Exception as e:
            parsed = {"function": func, "candidates": [], "top_k": []}

        # Fallback: if empty, prefer an overlapping candidate; else top-1
        if not parsed.get("top_k"):
            chosen = None
            if must_pick:
                for i, r in enumerate(ranked[:3]):
                    if has_overlap(r):
                        chosen = i
                        break
            if chosen is None:
                chosen = 0
            parsed = {"function": func, "candidates": [], "top_k": [chosen], "note": "fallback"}

        out_rows.append({
            "sentence_idx": sid,
            "function": func,
            "sentence_text": sent_text,
            "judgment": parsed,
            "num_candidates": len(ranked),
            "created_at": datetime.utcnow().isoformat()
        })

    return out_rows

# ---------------------------
# Merge final selections
# ---------------------------
def merge_selected(topk_rows, judgments):
    by_sid = defaultdict(list)
    for r in topk_rows:
        if r.get("rank") is not None:
            by_sid[int(r["sentence_idx"])].append(r)
    # order by rank
    for sid in by_sid:
        by_sid[sid] = sorted(by_sid[sid], key=lambda x: int(x.get("rank", 1_000_000)))

    final_rows = []
    for j in judgments:
        sid   = j["sentence_idx"]
        func  = j["function"]
        sent  = j["sentence_text"]
        topk_ids = j["judgment"].get("top_k", [])
        picked = []
        for kid in topk_ids:
            kid = int(kid)
            if 0 <= kid < len(by_sid[sid]):
                picked.append(by_sid[sid][kid])
        source = "LLM" if j["judgment"].get("candidates") else ("fallback" if topk_ids else "none")
        primary = picked[0] if picked else (by_sid[sid][0] if by_sid[sid] else {})
        alternates = picked[1:] if len(picked) > 1 else []

        final_rows.append({
            "sentence_idx": sid,
            "function": func,
            "sentence_text": sent,
            "selected_title": primary.get("title",""),
            "selected_year": primary.get("year",""),
            "selected_authors": primary.get("authors",""),
            "selected_arxiv_id": primary.get("arxiv_id",""),
            "selected_cosine": primary.get("cosine",""),
            "selection_source": source,  # "LLM" or "fallback"
            "alternates_count": len(alternates),
            "alternates_titles": [a.get("title","") for a in alternates],
        })
    return final_rows

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    assert os.path.exists(TOPK_PATH), f"Missing {TOPK_PATH}. Run retrieval first."
    topk_rows = read_jsonl(TOPK_PATH)

    judgments = judge_candidates(topk_rows, per_sentence_limit=PER_SENTENCE_LIMIT)
    write_jsonl(OUT_JSONL, judgments)

    # CSV summary (one row per sentence)
    summary_rows = []
    for r in judgments:
        summary_rows.append({
            "sentence_idx": r["sentence_idx"],
            "function": r["function"],
            "num_candidates": r["num_candidates"],
            "top_k_ids": ",".join(map(str, r["judgment"].get("top_k", []))),
            "source": "LLM" if r["judgment"].get("candidates") else ("fallback" if r["judgment"].get("top_k") else "none"),
            "sentence_text": r["sentence_text"][:160].replace("\n"," "),
        })
    pd.DataFrame(summary_rows).to_csv(OUT_CSV, index=False)

    # Final selected paper per sentence
    final_rows = merge_selected(topk_rows, judgments)
    write_jsonl(FINAL_JSONL, final_rows)
    pd.DataFrame(final_rows).to_csv(FINAL_CSV, index=False)

    print("✅ Wrote:")
    print("-", OUT_JSONL)
    print("-", OUT_CSV)
    print("-", FINAL_JSONL)
    print("-", FINAL_CSV)
