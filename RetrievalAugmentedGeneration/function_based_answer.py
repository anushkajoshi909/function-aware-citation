#!/usr/bin/env python3
# function_based_answer.py — function-aware selection with abstract-only synthesis
# Outputs:
#  - outputs/function_selection_results.json
#  - outputs/function_selection_results.jsonl
#  - outputs/function_answer_unified.txt   <-- unified 1–2 sentence answer covering all functions

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
# We set the flag to false because this prompt was really strict.
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
# This prompt was later discarded as its not generating answers like ti should.
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
    "Background": [
        "we present", "we describe", "we report", "we provide",
        "here we", "this paper presents", "this study presents",
        "we characterize", "we observe", "we measure",
        "we review", "overview of", "survey of",
        "is defined as", "definition of", "background on",
        "we derive (background",
    ],
    "Uses": [
        "we use", "we utilize", "we apply", "we employ", "we adopt",
        "we implement", "we exploit", "we make use of",
        "using the", "based on the", "leverag",
        "built on", "built upon", "with the aid of",
        "fine-tun", "calibrat using", "evaluat using", "train(ed) on",
        "we run on", "we run with", "we process with",
        "we follow the method of", "following the protocol of",
    ],
    "Compares": [
        "we compare", "compared with", "compared to", "comparison with",
        "head-to-head", "side-by-side", "versus", "vs.", "vs ",
        "in contrast to", "relative to", "as compared to",
        "benchmark against", "benchmarked against",
        "outperform", "performs better than", "beats",
        "superior to", "inferior to", "on par with",
        "statistically significant(ly) better", "non-inferior to",
        "ablation study", "ablation analyses",
        "comparison of methods", "comparative analysis",
    ],
    "Motivation": [
        "however,", "nevertheless,", "nonetheless,", "yet,", "but ",
        "remains unclear", "poorly understood", "little is known",
        "lack of", "limited understanding", "scarce data",
        "gap in the literature", "open problem", "open question",
        "challenge", "bottleneck", "shortcoming", "limitation",
        "conflict with", "inconsisten", "controvers",
        "need for", "necessitates", "calls for", "warrants",
        "motivat", "motivates", "to address this", "to fill this gap",
        "we aim to", "we seek to", "we intend to",
    ],
    "Extends": [
        "we extend", "extends", "extended", "extension of",
        "we generalize", "generalizes", "broaden", "wider class of",
        "we improve", "improves upon", "we refine", "we enhance",
        "we augment", "we advance", "build on", "build upon",
        "we introduce a new", "we propose a new", "novel method",
        "we modify", "we relax the assumption", "we unify",
        "more general formulation", "more general framework",
        "enables calculation of", "allows calculation of",
        "achieves higher accuracy", "achieves better performance",
        "permits higher accuracy", "reduces complexity",
        "extends to", "generalizes to", "applicable to broader",
        "pushes beyond", "state-of-the-art results",
    ],
    "FutureWork": [
        "future work", "in future work", "further work",
        "we plan to", "we intend to", "we will investigate",
        "we are exploring", "we will explore",
        "left for future", "beyond the scope", "outside the scope",
        "requires further study", "warrants further investigation",
        "remains to be seen", "remains open", "open avenue",
        "to be addressed", "to be studied", "to be explored",
        "we will release", "we will make available",
    ],
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

# ========= Scoring helpers (kept, but NOT used for ordering top-K) =========
def cosine_of(c: dict) -> float:
    try:
        return float(c.get("cosine", c.get("score", 0.0)) or 0.0)
    except Exception:
        return 0.0

def core_hits_in_text(txt: str, core_terms: List[str]) -> int:
    t = (txt or "").lower()
    return sum(1 for ct in set(x.lower() for x in core_terms or []) if ct in t)

def has_func_cue_in_text(txt: str, func: str) -> bool:
    return has_function_cue(txt, normalize_function(func))

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
def verify_candidate_for_function(query: str, func: str, cand: Dict[str, Any],
                                  core_terms: List[str], debug: bool=False, strict: bool=True) -> Verdict:
    # We will always call this with strict=False from the selector.
    func_norm = normalize_function(func)
    abs_txt = cand.get("abstract_full") or cand.get("abstract") or ""
    pid_cand = cand.get("paper_id") or cand.get("arxiv_id") or cand.get("id") or "unknown"

    # optional soft floor in strict mode (kept for completeness; not used when strict=False)
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

# ========= NEW: Unified synthesis across multiple functions =========
def synthesize_multi_function_answer(query: str,
                                     func_winners: Dict[str, Dict[str, Any]]) -> str:
    """
    Produce a single 1–2 sentence coherent answer that covers ALL functions.
    Each clause should be tagged with [Function] (e.g., [Extends], [Motivation]).
    Use only info from the provided TITLE/ABSTRACT snippets (no invention).
    If multiple papers are involved, cite (paper_id) inline before the period.
    """
    packs = []
    for f, w in func_winners.items():
        if not w:
            continue
        packs.append({
            "function": f,
            "paper_id": w.get("paper_id",""),
            "title": w.get("title",""),
            "abstract": w.get("abstract","")
        })

    if not packs:
        return "No sufficient evidence to answer coherently across the requested functions."

    functions_str = ", ".join([p["function"] for p in packs])
    evidence_blocks = []
    for p in packs:
        evidence_blocks.append(
            f"[{p['function']}] PAPER {p['paper_id']}\nTITLE: {cand_safe(p['title'], 500)}\nABSTRACT: {cand_safe(p['abstract'], 3000)}"
        )
    evidence_text = "\n\n".join(evidence_blocks)

    prompt = f"""Write ONE coherent answer in 1–2 sentences to the QUERY, covering ALL of these functions: {functions_str}.
Rules:
- Use ONLY the supplied titles/abstracts (no invented facts).
- Paraphrase (do not quote verbatim).
- Tag the clause for each function with [Function] (e.g., ... [Extends]).
- If a clause is supported by a specific paper, append the paper id in parentheses immediately before the period, e.g., ... (astro-ph/0103184).
- Stay concise; 1–2 sentences total.
- No preface, bullets, or extra text.

QUERY: {cand_safe(query, 800)}

EVIDENCE:
{evidence_text}
"""

    ans = llm_text(prompt, model=DEFAULT_MODEL, temperature=0.15, max_tokens=180).strip()
    # Enforce max of 2 sentences
    parts = re.split(r"(?<=\.)\s+", ans)
    if len(parts) > 2:
        ans = " ".join(parts[:2]).strip()
    return ans

# ========= Selection per function (RELAXED-ONLY + TOP-K BY RANK) =========
def pick_and_answer_for_function(query: str, func: str, candidates: List[Dict[str, Any]],
                                 core_terms: List[str], max_check: int, debug: bool=False,
                                 suppress_restate: bool=False) -> Dict[str, Any]:
    dbg(debug, "select | function=", func, "| candidates_in=", len(candidates))
    dbg(debug, "select | core_terms=", core_terms)

    # ---- NO PRUNING: keep all retrieved candidates ----
    cand_pool0 = candidates[:]  # consider everything we have
    dbg(debug, f"select | no cosine prune; kept={len(cand_pool0)}/{len(candidates)}")

    # Take top-N by original retrieval rank (ascending), preserving retrieval order.
    def _safe_rank(c):
        try:
            return int(c.get("rank", 1_000_000))
        except Exception:
            return 1_000_000
    cand_pool0_sorted = sorted(cand_pool0, key=_safe_rank)
    cand_pool = cand_pool0_sorted[:max_check]

    dbg(debug, "select | pool_size=", len(cand_pool),
        "| ranks=", [c.get("rank") for c in cand_pool],
        "| pids=", [c.get("paper_id") or c.get("arxiv_id") for c in cand_pool])

    # Single pass: relaxed verification only (as requested)
    verdicts: List[Verdict] = []
    for i, c in enumerate(cand_pool, 1):
        dbg(debug, f"select | verify relaxed {i}/{len(cand_pool)}")
        verdicts.append(verify_candidate_for_function(
            query, func, c, core_terms=core_terms, debug=debug, strict=False))

    supported = [v for v in verdicts if v.supports and not v.invalid_reason]
    supported.sort(key=lambda v: (v.fit + (0.03 if v.cue_hint else 0.0),
                                  v.topicality, v.retrieval_score),
                   reverse=True)
    winner = supported[0] if supported else None

    # ===== Arbiter with fallback to next supported candidate =====
    if winner:
        if not arbiter_on_topic(query, core_terms, winner.title, winner.abstract):
            dbg(debug, "arbiter | rejected off-topic winner:", winner.paper_id)
            winner = None
            # Try remaining supported candidates in order
            for alt in supported[1:]:
                if arbiter_on_topic(query, core_terms, alt.title, alt.abstract):
                    dbg(debug, "arbiter | accepted fallback:", alt.paper_id)
                    winner = alt
                    break
                else:
                    dbg(debug, "arbiter | rejected fallback:", alt.paper_id)

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
    ap.add_argument("--max-check", type=int, default=10, help="Max candidates to verify per function.")
    ap.add_argument("--debug", action="store_true", help="Print debug logs.")
    args = ap.parse_args()
    debug = args.debug

    dbg(debug, "main | args:", args)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_pretty = os.path.join(OUT_DIR, "function_selection_results.json")
    out_jsonl  = os.path.join(OUT_DIR, "function_selection_results.jsonl")
    # removed per-function answers file

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

        processed_funcs.add(func)

    # Save originals
    with open(out_pretty, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for obj in bundle:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # ========= Unified multi-function answer synthesis =========
    # Build a function -> winner map with title + recovered abstract
    func_winners: Dict[str, Dict[str, Any]] = {}

    # Index candidates by paper_id and by rank for recovery
    by_pid = {}
    for c in cands:
        pid = c.get("paper_id") or c.get("arxiv_id") or c.get("id")
        if pid:
            by_pid[str(pid)] = c
    by_rank = {str(c.get("rank")): c for c in cands if c.get("rank") is not None}

    for obj in bundle:
        func = obj["function"]
        res = obj["result"]
        w = res.get("winner") if res else None
        if not w:
            continue
        # Recover full abstract from candidate pools
        abstract_full = ""
        pid = w.get("paper_id")
        rank = str(w.get("id"))
        cand = by_pid.get(str(pid)) or by_rank.get(rank)
        if cand:
            abstract_full = cand.get("abstract_full") or cand.get("abstract") or ""
        func_winners[func] = {
            "paper_id": w.get("paper_id",""),
            "title": w.get("title",""),
            "abstract": abstract_full
        }

    unified_answer = synthesize_multi_function_answer(query, func_winners)

    out_unified = os.path.join(OUT_DIR, "function_answer_unified.txt")
    with open(out_unified, "w", encoding="utf-8") as f:
        f.write(unified_answer.strip() + "\n")

    print("\n" + "="*80)
    print("UNIFIED ANSWER (1–2 sentences, tagged):")
    print(unified_answer)
    print(f"\n📄 Saved: {out_pretty}")
    print(f"📄 Saved: {out_jsonl}")
    print(f"📝 Unified answer: {out_unified}")

if __name__ == "__main__":
    main()
