#!/usr/bin/env python3
# function_based_final.py — query-only, function-aware selection with full abstracts + LLM answer synthesis
# Outputs:
#  - outputs/function_selection_results.json
#  - outputs/function_selection_results.jsonl
#  - outputs/function_answers.txt               <-- answers-only lines

import os
import re
import json
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Set
from string import Template

# ========= Fixed config =========
CLASSIFIED_PATH = "classified_outputs.jsonl"
TOPK_PATH = "/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/outputs/topk_candidates_query.jsonl"
OUT_DIR = "outputs"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
API_KEY_FILE = os.path.join(os.path.expanduser("~"), ".scadsai-api-key")

# ========= LLM client =========
try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("pip install openai")

def load_client():
    if not os.path.exists(API_KEY_FILE):
        raise SystemExit("❌ API key file not found at ~/.scadsai-api-key")
    with open(API_KEY_FILE, "r", encoding="utf-8") as f:
        key = f.read().strip()
    return OpenAI(base_url="https://llm.scads.ai/v1", api_key=key)

CLIENT = load_client()

def llm_chat(prompt_user: str, model: str = DEFAULT_MODEL, temperature: float = 0.0, max_tokens: int = 900) -> str:
    resp = CLIENT.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return STRICT JSON only. No prose, no markdown, no code fences."},
            {"role": "user", "content": prompt_user}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

def llm_text(prompt_user: str, model: str = DEFAULT_MODEL, temperature: float = 0.25, max_tokens: int = 180) -> str:
    resp = CLIENT.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Write a concise answer. No preface, no bullets, no extra text."},
            {"role": "user", "content": prompt_user}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

# ========= Small utils / debug =========
def ts() -> str:
    return time.strftime("[%H:%M:%S]")

def dbg(flag: bool, *args):
    if flag: print(ts(), *args, flush=True)

CANON_FUNCS = {
    "background": "Background",
    "uses": "Uses",
    "compares": "Compares",
    "motivation": "Motivation",
    "extends": "Extends",
    "futurework": "FutureWork",
    "future_work": "FutureWork",
    "future-work": "FutureWork",
}
ALL_FUNCS = ["Background","Uses","Compares","Motivation","Extends","FutureWork"]

def normalize_function(label: str) -> str:
    if not label:
        return "Background"
    return CANON_FUNCS.get(label.strip().lower(), "Background")

def parse_json_strict(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s = text.strip().strip("`").strip()
        start = s.find("{")
        if start == -1:
            return {"supports": False, "fit": 0, "topicality": 0, "quote": "", "why": "JSON parse failed"}
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i+1])
                    except Exception:
                        return {"supports": False, "fit": 0, "topicality": 0, "quote": "", "why": "JSON parse failed"}
        return {"supports": False, "fit": 0, "topicality": 0, "quote": "", "why": "JSON parse failed"}

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def evidence_in_text(quote: str, title: str, abstract: str) -> bool:
    if not quote: return False
    q = normalize_space(quote).lower().strip(' "')
    text = normalize_space((title or "") + " " + (abstract or "")).lower()
    return q and (q in text)

def cand_safe(text: str, max_chars: int) -> str:
    if not text: return ""
    return text.replace("\u0000", " ").strip()[:max_chars]

# ========= Domain-term extraction for selection =========
STOP = set("""
a an and the of to in on for with by as at or but if than then from into over under between within without
about via per through across is are was were be been being have has had do does did can could may might
will would shall should must not no nor also both either neither this that these those it its their our your
his her them they we you i such thus there here where when which who whom whose what why how
""".split())

def content_terms_from_query(q: str, min_len=2):
    toks = re.findall(r"\b[\w-]+\b", (q or "").lower())
    return [t for t in toks if len(t) >= min_len and t not in STOP]

def overlap_score(cand: dict, terms: List[str]) -> int:
    hay = f"{cand.get('title','')} {(cand.get('abstract_full') or cand.get('abstract') or '')}".lower()
    return sum(1 for t in set(terms) if t in hay)

# ========= Prompts (Template-based) =========
FUNC_EVAL_PROMPT_TPL = Template("""You are a scientific function checker.

Given a user QUERY, a list of DOMAIN TERMS, and a target FUNCTION, decide whether the PAPER's
TITLE/ABSTRACT contain enough function-appropriate information to answer the QUERY.

Normalize FUNCTION to one of:
Background, Uses, Compares, Motivation, Extends, FutureWork.

STRICT RULES:
- Evidence MUST be a direct quote from TITLE or ABSTRACT (<= 40 words); no paraphrase.
- The quote MUST include at least one DOMAIN TERM (or a clear synonym of one).
- If function-specific support is missing, or DOMAIN TERMS are absent, set "supports": false and "fit": 0.0.

Return strict JSON:
{
  "supports": true/false,
  "fit": 0-1,
  "topicality": 0-1,
  "quote": "<<=40 words from TITLE or ABSTRACT>>",
  "why": "short reason tied to the function cue",
  "paper_id": "$PAPER_ID"
}

QUERY: $QUERY
DOMAIN TERMS: $TERMS
FUNCTION: $FUNCTION
TITLE: $TITLE
ABSTRACT: $ABSTRACT
""")

RELAXED_FUNC_EVAL_PROMPT_TPL = Template("""You are a scientific function checker.

Given a user QUERY and a target FUNCTION, decide whether the PAPER's TITLE/ABSTRACT
contain enough function-appropriate information to answer the QUERY.

Normalize FUNCTION to one of:
Background, Uses, Compares, Motivation, Extends, FutureWork.

STRICT RULES:
- Evidence MUST be a direct quote from TITLE or ABSTRACT (<= 40 words); no paraphrase.
- Prefer a quote containing a relevant term from the QUERY, but if none exists, choose the best function-relevant quote anyway.

Return strict JSON:
Due to JSON-only requirement, return exactly:
{
  "supports": true/false,
  "fit": 0-1,
  "topicality": 0-1,
  "quote": "<<=40 words from TITLE or ABSTRACT>>",
  "why": "short reason tied to the function cue",
  "paper_id": "$PAPER_ID"
}

QUERY: $QUERY
FUNCTION: $FUNCTION
TITLE: $TITLE
ABSTRACT: $ABSTRACT
""")

# ---- Single unified answer template for all functions ----
ANSWER_TPL = Template("""Answer the user's QUERY using ONLY the PAPER TITLE/ABSTRACT below.
Write 1–2 concise sentences that satisfy the FUNCTION.
- No invented facts. Do not quote verbatim; paraphrase.
- $FUNC_INSTR
$AVOID_RESTATE_INSTR

QUERY: $QUERY
FUNCTION: $FUNCTION
PAPER TITLE: $TITLE
PAPER ABSTRACT: $ABSTRACT
""")

# Function-specific instruction lines inserted into the unified template
FUNCTION_INSTRUCTIONS = {
    "Background": "State only the empirical/definitional facts relevant to the query; avoid causal or prescriptive language.",
    "Uses": "Describe how a known method/tool/dataset is applied in this work (who uses what for which purpose).",
    "Compares": "State what is being compared to what (methods/tools/datasets/results) and on what basis.",
    "Motivation": "State the problem/gap/tension that motivates the work and why it matters; avoid repeating background facts if already given.",
    "Extends": "State what prior method/result is extended and how (scope, accuracy, efficiency, assumptions).",
    "FutureWork": "State explicit future directions or open problems mentioned by the abstract.",
}

# ========= Evidence cue policies =========
# For each function, define cue words/phrases that must appear in the *evidence quote* to count as support.
FUNCTION_CUES = {
    "Background": [
        "we (measure|present|show|report)", "is measured", "we find", "we show", "we discuss", "we study",
        "is discussed", "is defined", "we analyze", "is analyzed", "we provide", "we identify", "is identified"
    ],
    "Uses": [
        "we use", "we utilize", "we apply", "using", "based on", "leverag", "adopt", "employ"
    ],
    "Compares": [
        "compare", "compared", "comparison", "versus", "vs.", "relative to", "contrast", "benchmark"
    ],
    "Motivation": [
        "challenge", "problem", "gap", "issue", "crisis", "tension", "discrepanc", "conflict", "need", "motivat"
    ],
    "Extends": [
        "extend", "extends", "extended", "generalize", "improve", "build on", "refine", "enhance"
    ],
    "FutureWork": [
        "future work", "we plan", "we will", "will investigate", "left for future", "beyond the scope",
        "in the future", "further study", "remains to", "open question", "to be addressed"
    ],
}

# Policy: require cue in quote for these functions
REQUIRE_CUE = {
    "Background": False,   # keep lenient; evidence + domain terms suffice
    "Uses": True,
    "Compares": True,
    "Motivation": True,
    "Extends": True,
    "FutureWork": True,
}

def has_function_cue(text: str, func_norm: str) -> bool:
    cues = FUNCTION_CUES.get(func_norm, [])
    t = (text or "").lower()
    for cue in cues:
        # treat patterns with regex metachar where appropriate
        if "(" in cue or "." in cue or "|" in cue:
            try:
                if re.search(cue, t):
                    return True
            except re.error:
                pass
        elif cue in t:
            return True
    return False

# ========= Data structures =========
@dataclass
class Verdict:
    supports: bool
    fit: float
    topicality: float
    quote: str
    why: str
    paper_id: str
    id: int
    title: str
    abstract: str
    authors: Any
    year: str
    retrieval_score: float = 0.0
    invalid_reason: str = ""

def clamp01(x) -> float:
    try: v = float(x)
    except Exception: v = 0.0
    return max(0.0, min(1.0, v))

def enforce_evidence(v: Verdict) -> Verdict:
    if not v.quote:
        v.supports = False; v.fit = 0.0; v.invalid_reason = "no_quote"; return v
    if not evidence_in_text(v.quote, v.title, v.abstract):
        v.supports = False; v.fit = 0.0; v.invalid_reason = "quote_not_in_abstract"; return v
    return v

# ========= IO helpers =========
def read_last_jsonl(path: str, debug=False) -> Dict[str, Any]:
    dbg(debug, "read_last_jsonl | path=", path)
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    dbg(debug, "read_last_jsonl | total_lines=", len(lines))
    if not lines:
        raise ValueError(f"No lines found in {path}")
    return json.loads(lines[-1])

def load_topk_jsonl(path: str, debug=False) -> List[Dict[str, Any]]:
    dbg(debug, "load_topk_jsonl | path=", path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            try:
                obj = json.loads(ln)
                rows.append(obj)
            except json.JSONDecodeError:
                continue
    dbg(debug, "load_topk_jsonl | rows=", len(rows))
    if rows:
        first = rows[0]
        abs_len = len((first.get("abstract_full") or first.get("abstract") or ""))
        dbg(debug, "load_topk_jsonl | first: rank=", first.get("rank"),
            "paper_id=", first.get("paper_id") or first.get("arxiv_id"),
            "title_snip='{}'".format((first.get("title","")[:70] + "…") if len(first.get("title",""))>70 else first.get("title","")),
            "abs_len=", abs_len)
    return rows

# ========= Core: verify / extract / synthesize =========
def verify_candidate_for_function(query: str, func: str, cand: Dict[str, Any],
                                  terms: List[str], debug: bool=False, strict: bool=True) -> Verdict:
    func_norm = normalize_function(func)
    abs_txt = cand.get("abstract_full") or cand.get("abstract") or ""
    pid_cand = cand.get("paper_id") or cand.get("arxiv_id") or cand.get("id") or "unknown"

    prompt_tpl = FUNC_EVAL_PROMPT_TPL if strict else RELAXED_FUNC_EVAL_PROMPT_TPL
    prompt = prompt_tpl.substitute(
        QUERY=cand_safe(query, 1500),
        TERMS=", ".join(sorted(set(terms)))[:300] if strict else "",
        FUNCTION=func_norm,
        TITLE=cand_safe(cand.get("title",""), 900),
        ABSTRACT=cand_safe(abs_txt, 8000),
        PAPER_ID=pid_cand
    )

    dbg(debug, "verify | func=", func_norm, "| rank=", cand.get("rank"),
        "| pid=", pid_cand, "| title_snip='{}'".format((cand.get("title","")[:70]+"…") if len(cand.get("title",""))>70 else cand.get("title","")),
        "| abs_len=", len(abs_txt))
    raw = llm_chat(prompt, model=DEFAULT_MODEL, temperature=0.0, max_tokens=900)
    obj = parse_json_strict(raw)

    pid_resp = str(obj.get("paper_id", "")).strip()
    pid_final = pid_resp if pid_resp and pid_resp not in {"{PAPER_ID}", "unknown"} else pid_cand

    v = Verdict(
        supports=bool(obj.get("supports", False)),
        fit=clamp01(obj.get("fit", 0)),
        topicality=clamp01(obj.get("topicality", 0)),
        quote=str(obj.get("quote", ""))[:500],
        why=str(obj.get("why", ""))[:300],
        paper_id=pid_final,
        id=int(cand.get("rank", 0) or 0),
        title=cand.get("title", ""),
        abstract=abs_txt,
        authors=cand.get("authors", ""),
        year=str(cand.get("year", "")),
        retrieval_score=float(cand.get("score") or c.get("cosine") or 0.0) if (c:=cand) else 0.0
    )
    v = enforce_evidence(v)

    # Extra guard: enforce function-specific cue presence where required
    if v.supports and REQUIRE_CUE.get(func_norm, False):
        if not has_function_cue(v.quote, func_norm):
            v.supports = False
            v.fit = 0.0
            v.invalid_reason = "missing_function_cue"

    dbg(debug, "verify | result supports=", v.supports, "fit=", f"{v.fit:.2f}",
        "top=", f"{v.topicality:.2f}", "invalid=", v.invalid_reason or "—")
    return v

def synthesize_answer(query: str, func: str, title: str, abstract: str,
                      paper_id: str, suppress_restate: bool=False) -> str:
    func_norm = normalize_function(func)

    func_instr = FUNCTION_INSTRUCTIONS.get(func_norm, "")
    avoid_restate_instr = ""
    if func_norm == "Motivation" and suppress_restate:
        avoid_restate_instr = "- Do not restate background measurements; assume Background already provided them."

    prompt = ANSWER_TPL.substitute(
        QUERY=cand_safe(query, 1000),
        FUNCTION=func_norm,
        TITLE=cand_safe(title, 600),
        ABSTRACT=cand_safe(abstract, 8000),  # harmless splitter to avoid accidental Template var; next line is real:
        FUNC_INSTR=func_instr,
        AVOID_RESTATE_INSTR=avoid_restate_instr
    )

    ans = llm_text(prompt, model=DEFAULT_MODEL, temperature=0.15, max_tokens=180).strip()

    # Always end with (paper_id).
    if not ans.endswith(f"({paper_id})."):
        ans = re.sub(r"\s*\([^)]*\)\.?$", "", ans)  # strip any old () citation
        ans = ans.rstrip(".") + f" ({paper_id})."

    # Enforce 1–2 sentences max
    parts = re.split(r"(?<=\.)\s+", ans)
    if len(parts) > 2:
        ans = " ".join(parts[:2])

    # Keep Background strictly factual and one sentence
    if func_norm == "Background":
        first = re.split(r"(?<=\.)\s+", ans)[0].rstrip(".")
        ans = first + f" ({paper_id})."

    return ans

# ========= Selection per function =========
def pick_and_answer_for_function(query: str, func: str, candidates: List[Dict[str, Any]],
                                 max_check: int, debug: bool=False,
                                 suppress_restate: bool=False) -> Dict[str, Any]:
    terms = content_terms_from_query(query)
    dbg(debug, "select | function=", func, "| candidates_in=", len(candidates))
    dbg(debug, "select | domain_terms=", terms)

    # A) top by retrieval
    cand_by_retrieval = sorted(
        candidates, key=lambda c: float(c.get("score") or c.get("cosine") or 0.0), reverse=True
    )[:max_check]

    # B) overlap >=1
    scored = [(overlap_score(c, terms), c) for c in candidates]
    scored.sort(key=lambda t: (t[0], float(t[1].get("score") or t[1].get("cosine") or 0.0)), reverse=True)
    cand_by_overlap = [c for ov, c in scored if ov >= 1][:max_check]

    # Merge unique
    seen = set(); cand_pool = []
    for c in cand_by_overlap + cand_by_retrieval:
        key = c.get("paper_id") or c.get("arxiv_id") or c.get("title")
        if key in seen: continue
        seen.add(key); cand_pool.append(c)
        if len(cand_pool) >= max_check: break

    dbg(debug, "select | pool_size=", len(cand_pool),
        "| ranks=", [c.get("rank") for c in cand_pool],
        "| pids=", [c.get("paper_id") or c.get("arxiv_id") for c in cand_pool])

    # Pass 1: strict
    verdicts: List[Verdict] = []
    for i, c in enumerate(cand_pool, 1):
        dbg(debug, f"select | verify strict {i}/{len(cand_pool)}")
        verdicts.append(verify_candidate_for_function(query, func, c, terms, debug=debug, strict=True))

    supported = [v for v in verdicts if v.supports and not v.invalid_reason]
    supported.sort(key=lambda v: (v.fit, v.topicality, v.retrieval_score), reverse=True)
    winner = supported[0] if supported else None

    # Pass 2: relaxed if needed
    if winner is None:
        dbg(debug, "select | strict pass found 0 winners → trying relaxed")
        verdicts_relaxed: List[Verdict] = []
        K = min(6, len(cand_pool))
        for i, c in enumerate(cand_pool[:K], 1):
            dbg(debug, f"select | verify relaxed {i}/{K}")
            verdicts_relaxed.append(verify_candidate_for_function(query, func, c, terms, debug=debug, strict=False))
        verdicts.extend(verdicts_relaxed)
        supported2 = [v for v in verdicts_relaxed if v.supports and not v.invalid_reason]
        supported2.sort(key=lambda v: (v.fit, v.topicality, v.retrieval_score), reverse=True)
        winner = supported2[0] if supported2 else None

    # Build answer via LLM synthesis from the winner's abstract
    answer_sentence = ""
    if winner and winner.quote:
        answer_sentence = synthesize_answer(
            query, func, winner.title, winner.abstract,
            paper_id=winner.paper_id,
            suppress_restate=suppress_restate
        )

    # Bundle
    return {
        "function": normalize_function(func),
        "winner": ({
            "id": winner.id,
            "paper_id": winner.paper_id,
            "title": winner.title,
            "authors": winner.authors,
            "year": winner.year,
            "fit": round(winner.fit, 3),
            "topicality": round(winner.topicality, 3),
            "retrieval_score": round(winner.retrieval_score, 3),
            "quote": winner.quote,
            "why": winner.why
        } if winner else None),
        "answer_sentence": answer_sentence,
        "verdicts": [
            {"id": v.id, "paper_id": v.paper_id, "fit": round(v.fit,3),
             "topicality": round(v.topicality,3), "retrieval": round(v.retrieval_score,3),
             "supports": v.supports, "invalid_reason": v.invalid_reason}
            for v in verdicts
        ]
    }

# ========= Main =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-check", type=int, default=10, help="Max candidates to verify per function.")
    ap.add_argument("--debug", action="store_true", help="Print debug logs.")
    args = ap.parse_args()
    debug = args.debug

    dbg(debug, "main | args:", args)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_pretty = os.path.join(OUT_DIR, "function_selection_results.json")
    out_jsonl  = os.path.join(OUT_DIR, "function_selection_results.jsonl")
    out_answers = os.path.join(OUT_DIR, "function_answers.txt")

    rec = read_last_jsonl(CLASSIFIED_PATH, debug=debug)
    query = (rec.get("query") or "").strip()
    funcs = ((rec.get("citation_function_classification") or {}).get("citation_functions") or [])
    funcs = [normalize_function(f) for f in funcs if f] or ["Background"]
    # Keep only known functions, preserve order
    funcs = [f for f in funcs if f in ALL_FUNCS]
    dbg(debug, "main | query='{}' | funcs={}".format(query, funcs))

    cands = load_topk_jsonl(TOPK_PATH, debug=debug)
    if not cands:
        print(f"No candidates found in {TOPK_PATH}")
        return

    bundle = []
    answer_lines = []

    processed_funcs: Set[str] = set()

    for func in funcs:
        dbg(debug, "main | running function=", func)
        suppress_restate = (func == "Motivation" and "Background" in processed_funcs)

        result = pick_and_answer_for_function(
            query, func, cands, max_check=args.max_check, debug=debug,
            suppress_restate=suppress_restate
        )
        bundle.append({"query": query, "function": func, "result": result})

        print("\n" + "="*80)
        print(f"FUNCTION: {func}")
        print(f"QUERY   : {query}")
        if result["winner"]:
            print(f"- WINNER: {result['winner']['paper_id']} | {result['winner']['title']}")
            print(f"- QUOTE : {result['winner']['quote']}")
            print(f"- WHY   : {result['winner']['why']}")
            print(f"- ANSWER: {result['answer_sentence'] or '[no answer — insufficient concrete facts]'}")
            # answers-only: just answer + [Function] (no duplicate paper_id)
            ans = result["answer_sentence"] or ""
            tag_line = f"{ans} [{func}]".strip()
            answer_lines.append(tag_line)
        else:
            print("- No supporting candidate with sufficient evidence.")

        processed_funcs.add(func)

    with open(out_pretty, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for obj in bundle:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    with open(out_answers, "w", encoding="utf-8") as f:
        for line in answer_lines:
            f.write(line.strip() + "\n")

    print(f"\n📄 Saved: {out_pretty}")
    print(f"📄 Saved: {out_jsonl}")
    print(f"📝 Answers-only: {out_answers}")

if __name__ == "__main__":
    main()
