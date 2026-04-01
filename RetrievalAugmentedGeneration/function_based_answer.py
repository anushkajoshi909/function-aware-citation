import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any
# -------------------------
# OpenAI-compatible client (SCADS endpoint)
# -------------------------
try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("pip install openai")

API_KEY_PATH = os.path.join(os.path.expanduser("~"), ".scadsai-api-key")
if not os.path.exists(API_KEY_PATH):
    raise SystemExit("❌ API key file not found at ~/.scadsai-api-key")
with open(API_KEY_PATH, "r", encoding="utf-8") as f:
    API_KEY = f.read().strip()

CLIENT = OpenAI(base_url="https://llm.scads.ai/v1", api_key=API_KEY)
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


# =========================
# OpenAI-compatible client (SCADS endpoint)
# =========================
try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("pip install openai")

def load_client():
    api_key_path = os.path.join(os.path.expanduser("~"), ".scadsai-api-key")
    if not os.path.exists(api_key_path):
        raise SystemExit("❌ API key file not found at ~/.scadsai-api-key")
    with open(api_key_path, "r", encoding="utf-8") as f:
        api_key = f.read().strip()
    return OpenAI(base_url="https://llm.scads.ai/v1", api_key=api_key)

CLIENT = load_client()
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

def llm_chat(prompt_user: str,
             model: str = DEFAULT_MODEL,
             temperature: float = 0.0,
             max_tokens: int = 900) -> str:
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

def llm_text(prompt_user: str,
             model: str = DEFAULT_MODEL,
             temperature: float = 0.2,
             max_tokens: int = 120) -> str:
    """For generating the final function-aware answer sentence."""
    resp = CLIENT.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Output ONE well-formed sentence. No preface, no bullets, no extra text."},
            {"role": "user", "content": prompt_user}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

# =========================
# Utilities
# =========================
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
def normalize_function(label: str) -> str:
    if not label:
        return "Background"
    return CANON_FUNCS.get(label.strip().lower(), "Background")

def extract_json_block(text: str) -> str:
    s = text.strip().strip("`").strip()
    start = s.find("{")
    if start == -1:
        raise ValueError("No opening brace in response.")
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{": depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    raise ValueError("Unbalanced braces in response.")

def parse_json_strict(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return json.loads(extract_json_block(text))
        except Exception:
            return {"supports": False, "fit": 0, "topicality": 0, "quote": "", "why": "JSON parse failed"}

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def evidence_in_text(quote: str, title: str, abstract: str) -> bool:
    if not quote:
        return False
    q = normalize_space(quote).lower().strip(' "')
    text = normalize_space((title or "") + " " + (abstract or "")).lower()
    return q and (q in text)

def first_author(authors_str: str) -> str:
    if not authors_str: return ""
    parts = [a.strip() for a in re.split(r"[;,]| and ", authors_str) if a.strip()]
    if not parts: parts = [authors_str.strip()]
    cand = parts[0]
    tokens = cand.split()
    return tokens[-1] if tokens else cand

def cand_safe(text: str, max_chars: int) -> str:
    if not text: return ""
    return text.replace("\u0000", " ").strip()[:max_chars]

# Domain terms from query + overlap scoring
STOP = set("""
a an and the of to in on for with by as at or but if than then from into over under between within without
about via per through across is are was were be been being have has had do does did can could may might
will would shall should must not no nor also both either neither this that these those it its their our your
his her them they we you i such thus there here where when which who whom whose what why how
""".split())

def content_terms_from_query(q: str, min_len=2):
    toks = re.findall(r"\b[\w-]+\b", q.lower())
    return [t for t in toks if len(t) >= min_len and t not in STOP]

def overlap_score(cand: dict, terms: List[str]) -> int:
    hay = f"{cand.get('title','')} {cand.get('abstract','')}".lower()
    return sum(1 for t in set(terms) if t in hay)

# =========================
# Prompts
# =========================
FUNC_EVAL_PROMPT = """You are a scientific function checker.

Given a user QUERY, a list of DOMAIN TERMS, and a target FUNCTION, decide whether the PAPER's
TITLE/ABSTRACT contain enough function-appropriate information to answer the QUERY.

Normalize FUNCTION to one of:
Background, Uses, Compares, Motivation, Extends, FutureWork.

STRICT RULES:
- Evidence MUST be a direct quote from TITLE or ABSTRACT (<= 40 words); no paraphrase.
- The quote MUST include at least one DOMAIN TERM (or a clear synonym of one).
- If function-specific support is missing, or DOMAIN TERMS are absent, set "supports": false and "fit": 0.0.

Return strict JSON:
{{
  "supports": true/false,
  "fit": 0-1,
  "topicality": 0-1,
  "quote": "<<=40 words from TITLE or ABSTRACT>>",
  "why": "short reason tied to the function cue",
  "paper_id": "{PAPER_ID}"
}}

QUERY: {QUERY}
DOMAIN TERMS: {TERMS}
FUNCTION: {FUNCTION}
TITLE: {TITLE}
ABSTRACT: {ABSTRACT}
"""


STRUCT_PROMPT = """Extract concrete facts from the PAPER's TITLE/ABSTRACT that can answer the QUERY
for the given FUNCTION. Use exact spans; no outside info.

Return STRICT JSON:
{{
  "system": "entities/systems studied (e.g., Cs, Fr, Ba II, Ra II)",
  "method": "method or approach used",
  "baseline_or_previous": "baseline/previous methods named ('' if none)",
  "result_or_claim": "specific comparative result or extension claim",
  "evidence": [
     {{"span": "<<=20 words exact quote>>"}},
     {{"span": "<<=20 words exact quote>>"}}
  ]
}}

QUERY: {QUERY}
FUNCTION: {FUNCTION}
TITLE: {TITLE}
ABSTRACT: {ABSTRACT}
"""


CITE_ANSWER_PROMPT = """Write ONE formal sentence (<=40 words) that answers the QUERY in a way
that clearly serves the FUNCTION, using ONLY the PAPER's TITLE/ABSTRACT (no outside info).
Append a citation in parentheses with FIRST_AUTHOR LASTNAME and YEAR, like (Lastname YEAR).
Do NOT add any extra text.

QUERY: {QUERY}
FUNCTION: {FUNCTION}
PAPER TITLE: {TITLE}
PAPER ABSTRACT: {ABSTRACT}
FIRST AUTHOR: {FIRST_AUTHOR}
YEAR: {YEAR}
"""

# =========================
# Data structures
# =========================
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
    authors: str
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

# =========================
# IO helpers
# =========================
def read_last_jsonl(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise ValueError(f"No lines found in {path}")
    return json.loads(lines[-1])

def load_topk_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            try: rows.append(json.loads(ln))
            except json.JSONDecodeError: continue
    return rows

# =========================
# Core: verify / extract / build
# =========================
def verify_candidate_for_function(query: str, func: str, cand: Dict[str, Any], terms: List[str]) -> Verdict:
    func_norm = normalize_function(func)
    prompt = FUNC_EVAL_PROMPT.format(
        QUERY=cand_safe(query, 1500),
        TERMS=", ".join(sorted(set(terms)))[:300],
        FUNCTION=func_norm,
        TITLE=cand_safe(cand.get("title",""), 900),
        ABSTRACT=cand_safe(cand.get("abstract",""), 2500),
        PAPER_ID=cand.get("paper_id") or cand.get("arxiv_id") or ""
    )
    raw = llm_chat(prompt, model=DEFAULT_MODEL, temperature=0.0, max_tokens=900)
    obj = parse_json_strict(raw)

    v = Verdict(
        supports=bool(obj.get("supports", False)),
        fit=clamp01(obj.get("fit", 0)),
        topicality=clamp01(obj.get("topicality", 0)),
        quote=str(obj.get("quote", ""))[:500],
        why=str(obj.get("why", ""))[:300],
        paper_id=str(obj.get("paper_id", cand.get("paper_id",""))),
        id=int(cand.get("rank", 0) or 0),
        title=cand.get("title", ""),
        abstract=cand.get("abstract", ""),
        authors=cand.get("authors", ""),
        year=str(cand.get("year", "")),
        retrieval_score=float(cand.get("score") or cand.get("cosine") or 0.0)
    )
    v = enforce_evidence(v)
    return v

def extract_facts(query: str, func: str, title: str, abstract: str) -> Dict[str, Any]:
    raw = llm_chat(STRUCT_PROMPT.format(
        QUERY=cand_safe(query, 800),
        FUNCTION=normalize_function(func),
        TITLE=cand_safe(title, 600),
        ABSTRACT=cand_safe(abstract, 2500)
    ), model=DEFAULT_MODEL, temperature=0.0, max_tokens=500)
    obj = parse_json_strict(raw)
    # minimal shape
    obj.setdefault("system","")
    obj.setdefault("method","")
    obj.setdefault("baseline_or_previous","")
    obj.setdefault("result_or_claim","")
    obj.setdefault("evidence",[])
    return obj

def build_answer_from_facts(func: str, facts: dict, authors: str, year: str) -> str:
    last = first_author(authors)
    yr = (re.findall(r"\d{4}", year) or [""])[0]

    sys = facts.get("system","").strip()
    meth = facts.get("method","").strip()
    base = facts.get("baseline_or_previous","").strip()
    claim = facts.get("result_or_claim","").strip()

    sys_part = f" for {sys}" if sys else ""
    base_part = f" vs {base}" if base else ""

    fn = normalize_function(func)
    if fn == "Compares":
        if not claim:
            return ""
        # steer wording towards concrete claim
        verb = "outperforms" if re.search(r"\boutperform|\bsuperior|better\b", claim, re.I) else "shows"
        return f"{meth or 'The hybrid approach'}{sys_part} {verb} {claim}{base_part} ({last} {yr})."

    if fn == "Extends":
        if not claim and not meth:
            return ""
        return f"{meth or 'The approach'}{sys_part} extends prior calculations by {claim or 'broadening the scope to additional systems'} ({last} {yr})."

    if claim:
        return f"{meth or 'The approach'}{sys_part} {claim} ({last} {yr})."
    return ""

# =========================
# Selection per function
# =========================
def pick_and_answer_for_function(query: str, func: str, candidates: List[Dict[str, Any]], max_check: int = 20):
    terms = content_terms_from_query(query)
    # topic filter first
    scored = []
    for c in candidates:
        ov = overlap_score(c, terms)
        scored.append((ov, c))
    scored.sort(key=lambda t: (t[0], t[1].get("score",0.0)), reverse=True)
    # prefer candidates mentioning >=2 domain terms, else fall back to top few
    cand_pool = [c for ov,c in scored if ov >= 2][:max_check] or [c for ov,c in scored][:min(max_check, 8)]

    # verify candidates (enforces evidence + domain terms)
    verdicts: List[Verdict] = [verify_candidate_for_function(query, func, c, terms) for c in cand_pool]

    supported = [v for v in verdicts if v.supports and not v.invalid_reason]
    supported.sort(key=lambda v: (v.fit, v.topicality, v.retrieval_score), reverse=True)
    winner = supported[0] if supported else None

    # build factual answer only if we have a verified winner with evidence
    answer_sentence, facts = "", {}
    if winner and winner.quote:
        facts = extract_facts(query, func, winner.title, winner.abstract)
        answer_sentence = build_answer_from_facts(func, facts, winner.authors, winner.year)

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
        "facts": facts,
        "verdicts": [
            {"id": v.id, "paper_id": v.paper_id, "fit": round(v.fit,3),
             "topicality": round(v.topicality,3), "retrieval": round(v.retrieval_score,3),
             "supports": v.supports, "invalid_reason": v.invalid_reason}
            for v in verdicts
        ]
    }

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classified", default="classified_outputs.jsonl",
                    help="Path to the JSONL with the latest {query, citation_function_classification}.")
    ap.add_argument("--topk", default="/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/outputs/topk_candidates_query.jsonl",
                    help="Path to the JSONL with top-k candidates for the query.")
    ap.add_argument("--outdir", default="outputs", help="Directory to write results.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="LLM model id to use.")
    ap.add_argument("--max-check", type=int, default=20, help="Max candidates to verify per function.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out_pretty = os.path.join(args.outdir, "function_selection_results.json")
    out_jsonl  = os.path.join(args.outdir, "function_selection_results.jsonl")

    # load latest query + functions
    rec = read_last_jsonl(args.classified)
    query = (rec.get("query") or "").strip()
    funcs = ((rec.get("citation_function_classification") or {}).get("citation_functions") or [])
    funcs = [normalize_function(f) for f in funcs if f] or ["Background"]

    # load top-k candidates (same pool for each function)
    cands = load_topk_jsonl(args.topk)
    if not cands:
        print(f"No candidates found in {args.topk}")
        return

    bundle = []
    for func in funcs:
        result = pick_and_answer_for_function(query, func, cands, max_check=args.max_check)
        bundle.append({"query": query, "function": func, "result": result})

        # console preview
        print("\n" + "="*80)
        print(f"FUNCTION: {func}")
        print(f"QUERY   : {query}")
        if result["winner"]:
            print(f"- WINNER: {result['winner']['paper_id']} | {result['winner']['title']}")
            print(f"- QUOTE : {result['winner']['quote']}")
            print(f"- WHY   : {result['winner']['why']}")
            print(f"- ANSWER: {result['answer_sentence'] or '[no answer — insufficient concrete facts]'}")
        else:
            print("- No supporting candidate with sufficient evidence.")

    # save results
    with open(out_pretty, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for obj in bundle:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"\n📄 Saved: {out_pretty}")
    print(f"📄 Saved: {out_jsonl}")

if __name__ == "__main__":
    main()
