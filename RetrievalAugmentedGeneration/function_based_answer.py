#!/usr/bin/env python3
# function_based_final.py — function-aware selection with abstract-only synthesis
# Outputs:
#  - outputs/function_selection_results.json
#  - outputs/function_selection_results.jsonl
#  - outputs/function_answers.txt   <-- answers-only lines

import os
import re
import json
import time
import argparse
import unicodedata
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

def sanitize_query(q: str) -> str:
    if not q: return ""
    q = q.strip()
    if q.endswith('"') and not q.startswith('"'): q = q[:-1]
    if q.startswith('"') and not q.endswith('"'): q = q[1:]
    return re.sub(r"\s+", " ", q)

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

def normalize_for_match(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower().strip(' "\'`')

def evidence_in_text(quote: str, title: str, abstract: str) -> bool:
    if not quote: return False
    q = normalize_for_match(quote)
    text = normalize_for_match((title or "") + " " + (abstract or ""))
    return q and (q in text)

def cand_safe(text: str, max_chars: int) -> str:
    if not text: return ""
    return text.replace("\u0000", " ").strip()[:max_chars]

# ========= Domain-term extraction fallback =========
STOP = set("""
a an and the of to in on for with by as at or but if than then from into over under between within without
about via per through across is are was were be been being have has had do does did can could may might
will would shall should must not no nor also both either neither this that these those it its their our your
his her them they we you i such thus there here where when which who whom whose what why how
""".split())

def content_terms_from_query(q: str, min_len=3):
    toks = re.findall(r"\b[\w-]+\b", (q or "").lower())
    return [t for t in toks if len(t) >= min_len and t not in STOP]

# ========= LLM-based CORE ENTITIES extraction =========
CORE_ENTITIES_PROMPT_TPL = Template("""Extract the core scientific entities and concepts that MUST appear (exactly or via clear synonyms) for a result to be on-topic for the QUERY.

Return strict JSON:
{
  "entities": ["...", "..."],
  "concepts": ["...", "..."]
}

Rules:
- Keep it short (<=6 total items total across both lists).
- Include symbols/abbreviations if present (e.g., "NG boson", "Nambu–Goldstone").
- Prefer specific technical terms; avoid generic words like "model", "method", "framework".
- If the query names a phenomenon, observable, transition, or amplitude (e.g., "Lorentz symmetry breaking", "6s-7s", "s-d"), include it as-is.

QUERY: $QUERY
""")

def extract_core_entities(query: str) -> Dict[str, List[str]]:
    raw = llm_chat(CORE_ENTITIES_PROMPT_TPL.substitute(QUERY=query),
                   model=DEFAULT_MODEL, temperature=0.0, max_tokens=200)
    obj = parse_json_strict(raw)
    ents = obj.get("entities") or []
    conc = obj.get("concepts") or []
    # de-dup preserve order
    seen=set(); keep=[]
    for x in (ents + conc):
        s = str(x).strip()
        if s and s.lower() not in seen:
            seen.add(s.lower()); keep.append(s)
    return {"core": keep[:6]}

def extract_special_core_from_query(query: str) -> List[str]:
    q = query or ""
    toks = re.findall(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)+|[A-Z]{2,}|\d+[A-Za-z]+|\bCs\b", q)
    seen=set(); kept=[]
    for t in toks:
        k=t.lower()
        if k not in seen:
            seen.add(k); kept.append(t)
    return kept

def abstract_mentions_core(abstract: str, core_terms: List[str]) -> bool:
    a = (abstract or "").lower()
    return any(t.lower() in a for t in (core_terms or []))

# ========= Function descriptions (semantic guidance) =========
CITATION_FUNCTION_DESCRIPTIONS = """Possible citation functions:
- Background: Introduces known facts, definitions, or prior knowledge to set context.
- Uses: Describes how a known method or tool is applied in the current work.
- Compares: Compares two or more methods, tools, datasets, or results.
- Motivation: Highlights a gap, problem, or challenge that motivates the research.
- Extends: Builds upon or improves an existing method, approach, or result.
- FutureWork: Outlines work that is planned but not yet done.
"""

# ========= Prompts (Template-based) =========
FUNC_EVAL_PROMPT_TPL = Template("""You are a scientific function checker.

Task: Decide if the PAPER's TITLE/ABSTRACT contain enough information to answer the QUERY **for the given FUNCTION**.
Use the function definitions below; judge semantically (do not rely only on keywords).

$FUNC_DEFS

STRICT RULES:
- Provide a direct quote from TITLE or ABSTRACT (<= 40 words).
- The quote must be copied VERBATIM (exact substring). Do not paraphrase.
- The quote MUST include at least one item (exact or clear synonym) from CORE ENTITIES.
- If the paper is off-topic (CORE ENTITIES absent in the quote), set "supports": false and "why": "off_topic".
- If you cannot find a suitable exact quote, set "supports": false.

Return strict JSON:
{
  "supports": true/false,
  "fit": 0-1,
  "topicality": 0-1,
  "quote": "<<=40 words from TITLE or ABSTRACT>>",
  "why": "off_topic | no_function_evidence | sufficient",
  "paper_id": "$PAPER_ID"
}

QUERY: $QUERY
CORE ENTITIES: $CORE
FUNCTION: $FUNCTION
TITLE: $TITLE
ABSTRACT: $ABSTRACT
""")

RELAXED_FUNC_EVAL_PROMPT_TPL = Template("""You are a scientific function checker.

Task: Decide if the PAPER's TITLE/ABSTRACT contain enough information to answer the QUERY **for the given FUNCTION**.
Use the function definitions below; judge semantically (do not rely only on keywords).

$FUNC_DEFS

Rules (relaxed but safe):
- Provide a direct quote from TITLE or ABSTRACT (<= 40 words).
- The quote must be copied VERBATIM (exact substring). Do not paraphrase.
- Prefer a quote that includes at least one CORE ENTITY; if no such quote exists, you may select the best function-relevant quote **only if** the ABSTRACT elsewhere clearly mentions at least one CORE ENTITY (exact or clear synonym), and your chosen quote is substantively tied to that context.
- If the paper appears off-topic (no CORE ENTITIES anywhere), set "supports": false ("off_topic").
- If you cannot find a suitable exact quote, set "supports": false.

Return strict JSON:
{
  "supports": true/false,
  "fit": 0-1,
  "topicality": 0-1,
  "quote": "<<=40 words from TITLE or ABSTRACT>>",
  "why": "off_topic | no_function_evidence | sufficient",
  "paper_id": "$PAPER_ID"
}

QUERY: $QUERY
CORE ENTITIES: $CORE
FUNCTION: $FUNCTION
TITLE: $TITLE
ABSTRACT: $ABSTRACT
""")

# ---- Unified answer template for all functions ----
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

FUNCTION_INSTRUCTIONS = {
    "Background": "State only the empirical/definitional facts relevant to the query; avoid causal or prescriptive language.",
    "Uses": "Describe how a known method/tool/dataset is applied in this work (who uses what for which purpose).",
    "Compares": "State what is being compared to what (methods/tools/datasets/results) and on what basis.",
    "Motivation": "State the problem/gap/tension that motivates the work and why it matters; avoid repeating background facts if already given.",
    "Extends": "State what prior method/result is extended and how (scope, accuracy, efficiency, assumptions).",
    "FutureWork": "State explicit future directions or open problems mentioned by the abstract.",
}

FUNCTION_CUES = {
    "Background": ["we present","we report","we find","we show","we study","is defined","we analyze","we provide","we identify","is identified"],
    "Uses": ["we use","we utilize","we apply","using","based on","leverage","adopt","employ"],
    "Compares": ["compare","compared","comparison","versus","vs.","relative to","contrast","benchmark","compares favorably","favorable comparison","in contrast to","unlike","as compared to","avoid cancellation","avoids cancellation","reduces cancellation","vs","versus"],
    "Motivation": ["challenge","problem","gap","issue","crisis","tension","discrepanc","conflict","need","motivat"],
    "Extends": ["extend","extends","extended","generalize","improve","build on","refine","enhance","extends to","generalizes to","allows calculation of","enables calculation of","achieves higher accuracy","permits higher accuracy","broadens","pushes beyond"],
    "FutureWork": ["future work","we plan","we will","will investigate","left for future","beyond the scope","in the future","further study","remains to","open question","to be addressed"],
}

def has_function_cue(text: str, func_norm: str) -> bool:
    cues = FUNCTION_CUES.get(func_norm, [])
    t = (text or "").lower()
    for cue in cues:
        if cue in t:
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
    evidence_ok: bool = False
    on_topic_ok: bool = False
    cue_hint: bool = False

def clamp01(x) -> float:
    try: v = float(x)
    except Exception: v = 0.0
    return max(0.0, min(1.0, v))

def evidence_in_text_enforce(v: Verdict) -> Verdict:
    if not v.quote:
        v.supports = False; v.fit = 0.0; v.invalid_reason = "no_quote"; return v
    if not evidence_in_text(v.quote, v.title, v.abstract):
        v.supports = False; v.fit = 0.0; v.invalid_reason = "quote_not_in_abstract"; return v
    v.evidence_ok = True
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

# ========= Scoring helpers (cosine + blended) =========
def cosine_of(c: dict) -> float:
    try:
        return float(c.get("cosine", c.get("score", 0.0)) or 0.0)
    except Exception:
        return 0.0

def dynamic_cosine_cut(cands: List[Dict[str, Any]], keep_frac: float = 0.6, min_floor: float = 0.25) -> float:
    vals = sorted([cosine_of(c) for c in cands], reverse=True)
    if not vals:
        return min_floor
    k = max(1, int(len(vals)*keep_frac))
    t = vals[k-1]
    return max(t, min_floor)

def core_hits_in_text(txt: str, core_terms: List[str]) -> int:
    t = (txt or "").lower()
    return sum(1 for ct in set(x.lower() for x in core_terms or []) if ct in t)

def has_func_cue_in_text(txt: str, func: str) -> bool:
    return has_function_cue(txt, normalize_function(func))

def quick_string_bonus(cand: dict, core_terms: List[str]) -> float:
    bonus = 0.0
    for fld in ("title_matches", "abs_matches"):
        s = (cand.get(fld) or "").lower()
        if any(ct.lower() in s for ct in core_terms or []):
            bonus += 0.02
    return min(bonus, 0.04)

def subject_bias(pid: str, core_terms: List[str]) -> float:
    pid = (pid or "").lower()
    key = " ".join(core_terms).lower()
    if any(k in key for k in ["pnc", "s-d", "6s-7s", "cs "]):
        if pid.startswith("physics/"):
            return 0.04
    return 0.0

def blended_relevance(cand: dict, core_terms: List[str], func: str) -> float:
    cos = cosine_of(cand)
    txt = f"{cand.get('title','')} {(cand.get('abstract_full') or cand.get('abstract') or '')}"
    ch = core_hits_in_text(txt, core_terms)
    cue = 1.0 if has_func_cue_in_text(txt, func) else 0.0
    w_cos, w_core, w_cue = 0.60, 0.35, 0.05
    core_bonus = min(ch, 3) / 3.0
    base = w_cos*cos + w_core*core_bonus + w_cue*cue + quick_string_bonus(cand, core_terms)
    if ch == 0:
        base -= 0.10
    base += subject_bias(cand.get("paper_id",""), core_terms)
    return base

# ========= Final topicality arbiter =========
ON_TOPIC_ARBITER_PROMPT_TPL = Template("""Return strict JSON:
{ "on_topic": true/false, "reason": "short" }

A paper is ON-TOPIC for the QUERY only if its TITLE/ABSTRACT clearly relate to the CORE ENTITIES (exactly or via obvious synonyms).

QUERY: $QUERY
CORE ENTITIES: $CORE
TITLE: $TITLE
ABSTRACT: $ABSTRACT
""")

def arbiter_on_topic(query: str, core_terms: List[str], title: str, abstract: str) -> bool:
    prompt = ON_TOPIC_ARBITER_PROMPT_TPL.substitute(
        QUERY=cand_safe(query, 800),
        CORE=", ".join(core_terms[:6]),
        TITLE=cand_safe(title, 600),
        ABSTRACT=cand_safe(abstract, 3000),
    )
    raw = llm_chat(prompt, model=DEFAULT_MODEL, temperature=0.0, max_tokens=120)
    obj = parse_json_strict(raw)
    return bool(obj.get("on_topic", False))

# ========= Core: verify / extract / synthesize =========
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
    evidence_ok: bool = False
    on_topic_ok: bool = False
    cue_hint: bool = False

def verify_candidate_for_function(query: str, func: str, cand: Dict[str, Any],
                                  core_terms: List[str], debug: bool=False, strict: bool=True) -> Verdict:
    func_norm = normalize_function(func)
    abs_txt = cand.get("abstract_full") or cand.get("abstract") or ""
    pid_cand = cand.get("paper_id") or cand.get("arxiv_id") or cand.get("id") or "unknown"

    # optional soft floor in strict mode (still counts as checked)
    cos = cosine_of(cand)
    if strict and cos < 0.2:
        return Verdict(False, 0.0, 0.0, "", "cosine_too_low", pid_cand,
                       int(cand.get("rank", 0) or 0), cand.get("title",""),
                       abs_txt, cand.get("authors",""), str(cand.get("year","")),
                       retrieval_score=cos)

    prompt_tpl = FUNC_EVAL_PROMPT_TPL if strict else RELAXED_FUNC_EVAL_PROMPT_TPL
    prompt = prompt_tpl.substitute(
        QUERY=cand_safe(query, 1500),
        CORE=", ".join(core_terms[:6]),
        FUNCTION=func_norm,
        TITLE=cand_safe(cand.get("title",""), 900),
        ABSTRACT=cand_safe(abs_txt, 8000),
        PAPER_ID=pid_cand,
        FUNC_DEFS=CITATION_FUNCTION_DESCRIPTIONS
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
        retrieval_score=cos
    )

    v = evidence_in_text_enforce(v)

    if v.supports:
        quote_has_core = any(t.lower() in (v.quote or "").lower() for t in (core_terms or []))
        abstract_has_core = abstract_mentions_core(abs_txt, core_terms)
        if strict:
            v.on_topic_ok = quote_has_core
            if not v.on_topic_ok:
                v.supports = False
                v.fit = 0.0
                v.invalid_reason = "missing_core_in_quote"
        else:
            v.on_topic_ok = (quote_has_core or abstract_has_core)
            if not v.on_topic_ok:
                v.supports = False
                v.fit = 0.0
                v.invalid_reason = "off_topic_no_core_in_abstract"

    v.cue_hint = has_function_cue(v.quote, func_norm)

    dbg(debug, "verify | result supports=", v.supports, "fit=", f"{v.fit:.2f}",
        "top=", f"{v.topicality:.2f}", "invalid=", v.invalid_reason or "—",
        "cue_hint=", v.cue_hint)
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
        ABSTRACT=cand_safe(abstract, 8000),
        FUNC_INSTR=func_instr,
        AVOID_RESTATE_INSTR=avoid_restate_instr
    )

    ans = llm_text(prompt, model=DEFAULT_MODEL, temperature=0.15, max_tokens=180).strip()

    if not ans.endswith(f"({paper_id})."):
        ans = re.sub(r"\s*\([^)]*\)\.?$", "", ans)
        ans = ans.rstrip(".") + f" ({paper_id})."

    parts = re.split(r"(?<=\.)\s+", ans)
    if len(parts) > 2:
        ans = " ".join(parts[:2])

    if func_norm == "Background":
        first = re.split(r"(?<=\.)\s+", ans)[0].rstrip(".")
        ans = first + f" ({paper_id})."

    return ans

# ========= Selection per function =========
def pick_and_answer_for_function(query: str, func: str, candidates: List[Dict[str, Any]],
                                 core_terms: List[str], max_check: int, debug: bool=False,
                                 suppress_restate: bool=False) -> Dict[str, Any]:
    dbg(debug, "select | function=", func, "| candidates_in=", len(candidates))
    dbg(debug, "select | core_terms=", core_terms)

    # ---- NO PRUNING: keep all retrieved candidates ----
    cand_pool0 = candidates[:]  # consider everything we have
    dbg(debug, f"select | no cosine prune; kept={len(cand_pool0)}/{len(candidates)}")

    # Rank all candidates by blended relevance, then take up to max_check
    ranked = sorted(cand_pool0, key=lambda c: blended_relevance(c, core_terms, func), reverse=True)
    cand_pool = ranked[:max_check]

    dbg(debug, "select | pool_size=", len(cand_pool),
        "| ranks=", [c.get("rank") for c in cand_pool],
        "| pids=", [c.get("paper_id") or c.get("arxiv_id") for c in cand_pool])

    # Pass 1: strict
    verdicts: List[Verdict] = []
    for i, c in enumerate(cand_pool, 1):
        dbg(debug, f"select | verify strict {i}/{len(cand_pool)}")
        verdicts.append(verify_candidate_for_function(query, func, c, core_terms=core_terms, debug=debug, strict=True))

    supported = [v for v in verdicts if v.supports and not v.invalid_reason]
    supported.sort(key=lambda v: (v.fit + (0.03 if v.cue_hint else 0.0), v.topicality, v.retrieval_score), reverse=True)
    winner = supported[0] if supported else None

    # Pass 2: relaxed if needed — check the full pool
    if winner is None:
        dbg(debug, "select | strict pass found 0 winners → trying relaxed")
        verdicts_relaxed: List[Verdict] = []
        K = len(cand_pool)
        for i, c in enumerate(cand_pool[:K], 1):
            dbg(debug, f"select | verify relaxed {i}/{K}")
            verdicts_relaxed.append(verify_candidate_for_function(query, func, c, core_terms=core_terms, debug=debug, strict=False))
        verdicts.extend(verdicts_relaxed)
        supported2 = [v for v in verdicts_relaxed if v.supports and not v.invalid_reason]
        supported2.sort(key=lambda v: (v.fit + (0.03 if v.cue_hint else 0.0), v.topicality, v.retrieval_score), reverse=True)
        winner = supported2[0] if supported2 else None

    # Arbiter
    if winner:
        if not arbiter_on_topic(query, core_terms, winner.title, winner.abstract):
            dbg(debug, "arbiter | rejected off-topic winner:", winner.paper_id)
            winner = None

    if winner and winner.quote:
        answer_sentence = synthesize_answer(
            query, func, winner.title, winner.abstract,
            paper_id=winner.paper_id,
            suppress_restate=suppress_restate
        )
    else:
        answer_sentence = f"The abstract does not provide information relevant to {normalize_function(func)}."

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
             "supports": v.supports, "invalid_reason": v.invalid_reason,
             "evidence_ok": v.evidence_ok, "on_topic_ok": v.on_topic_ok, "cue_hint": v.cue_hint}
            for v in verdicts
        ]
    }

# ========= Main =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-check", type=int, default=200, help="Max candidates to verify per function.")
    ap.add_argument("--debug", action="store_true", help="Print debug logs.")
    args = ap.parse_args()
    debug = args.debug

    dbg(debug, "main | args:", args)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_pretty = os.path.join(OUT_DIR, "function_selection_results.json")
    out_jsonl  = os.path.join(OUT_DIR, "function_selection_results.jsonl")
    out_answers = os.path.join(OUT_DIR, "function_answers.txt")

    rec = read_last_jsonl(CLASSIFIED_PATH, debug=debug)
    query = sanitize_query(rec.get("query") or "")
    funcs = ((rec.get("citation_function_classification") or {}).get("citation_functions") or [])
    funcs = [normalize_function(f) for f in funcs if f] or ["Background"]
    funcs = [f for f in funcs if f in ALL_FUNCS]
    dbg(debug, "main | query='{}' | funcs={}".format(query, funcs))

    # core terms
    core = extract_core_entities(query)
    special = extract_special_core_from_query(query)
    core_terms = (core.get("core", []) or []) + special
    seen=set(); core_terms = [t for t in core_terms if not (t.lower() in seen or seen.add(t.lower()))]
    if not core_terms:
        core_terms = content_terms_from_query(query)
    dbg(debug, "main | core_terms=", core_terms)

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
            query, func, cands, core_terms=core_terms, max_check=args.max_check, debug=debug,
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
        else:
            print("- No supporting candidate with sufficient evidence.")
        print(f"- ANSWER: {result['answer_sentence'] or '[no answer — insufficient concrete facts]'}")

        answer_lines.append(f"{result['answer_sentence']} [{func}]")
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
