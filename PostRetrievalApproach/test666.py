#!/usr/bin/env python3
# test666.py — verification + function-aware citation sentence
# - Uses strict JSON prompts with escaped braces
# - Validates quoted evidence is in TITLE/ABSTRACT (not the sentence)
# - Tries a retry with guidance if the evidence fails
# - Aggregates top-k per sentence and prints + saves pretty JSON
# - Generates one-sentence function-aware citation for the best match

import os
import json
import re
from collections import defaultdict
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
             max_tokens: int = 160) -> str:
    """For generating the final one-sentence citation text (not strict JSON)."""
    resp = CLIENT.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You produce a single, well-formed sentence. No preface, no bullets."},
            {"role": "user", "content": prompt_user}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

# -------------------------
# Normalization
# -------------------------
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

# -------------------------
# Robust JSON extraction
# -------------------------
def extract_json_block(text: str) -> str:
    s = text.strip().strip("`").strip()
    start = s.find("{")
    if start == -1:
        raise ValueError("No opening brace in response.")
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
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
            block = extract_json_block(text)
            return json.loads(block)
        except Exception:
            return {"supports": False, "score": 0, "topicality": 0, "quote": "", "why": "JSON parse failed"}

# -------------------------
# Verification prompt (TITLE or ABSTRACT allowed)
#   IMPORTANT: literal braces in the JSON schema are escaped as {{ }}
# -------------------------
MICRO_PROMPT = """You are a scientific function checker.

Normalize the requested function to one of:
Background, Uses, Compares, Motivation, Extends, FutureWork.

Function cues (synonyms allowed; examples of phrases to look for):
- Background: review, overview, introduce, describe, summarize, survey, tutorial, "we review", "an overview of"
- Uses: use, apply, implement, employ, "we use", "we apply", "we implement", "applied to", "experiment using"
- Compares: compare, comparison, versus, vs., relative to, benchmark, outperforms, "we compare", "compared with"
- Motivation: limitation, challenge, problem, gap, uncertainty, bottleneck, "cannot explain", "insufficient", "breaks down", "hard to"
- Extends: extend, extension, build upon, generalize, improve upon, augment, broaden, "we extend", "we generalize"
- FutureWork: future work, plan, prospects, directions, "should be explored", "we will investigate", "further research", "remains to be done", "open questions", "future bounds"

STRICT RULES:
- Your evidence MUST be a direct quote from the TITLE or ABSTRACT (<= 40 words).
- Do NOT paraphrase the SENTENCE. Do NOT copy from SENTENCE.
- If you cannot find function-specific evidence, set "supports": false and "score": 0.0.

Return strict JSON:
{{
  "supports": true/false,
  "score": 0-1,
  "topicality": 0-1,
  "quote": "<<=40 words from TITLE or ABSTRACT>>",
  "why": "short reason tied to the function cue",
  "paper_id": "{PAPER_ID}"
}}

SENTENCE: {SENTENCE}
FUNCTION: {FUNCTION}
TITLE: {TITLE}
ABSTRACT: {ABSTRACT}
"""

# -------------------------
# Generation prompt for the final one-sentence citation
# -------------------------
CITE_SENT_PROMPT = """Write ONE formal sentence (<=40 words) that supports the FUNCTION for the SENTENCE using the PAPER's TITLE/ABSTRACT context.
- Clearly serve the FUNCTION (e.g., Motivation = highlight limitation; FutureWork = suggest next steps).
- Mention FIRST_AUTHOR LASTNAME and YEAR in parentheses at the end, like (Lastname YEAR).
- Do NOT copy the SENTENCE; rephrase.
- No extra text.

FUNCTION: {FUNCTION}
SENTENCE: {SENTENCE}
PAPER TITLE: {TITLE}
PAPER ABSTRACT: {ABSTRACT}
FIRST AUTHOR: {FIRST_AUTHOR}
YEAR: {YEAR}
"""

# -------------------------
# Data structures
# -------------------------
@dataclass
class MicroVerdict:
    supports: bool
    score: float
    topicality: float
    quote: str
    why: str
    paper_id: str
    id: int
    title: str
    abstract: str
    authors: str
    year: str
    invalid_reason: str = ""
    retrieval_score: float = 0.0

def clamp01(x) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    return max(0.0, min(1.0, v))

def cand_safe(text: str, max_chars: int) -> str:
    if not text:
        return ""
    return text.replace("\u0000", " ").strip()[:max_chars]

# -------------------------
# Evidence validation
# -------------------------
def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def evidence_in_text(quote: str, title: str, abstract: str) -> bool:
    if not quote:
        return False
    q = normalize_space(quote).lower().strip(' "')
    text = normalize_space((title or "") + " " + (abstract or "")).lower()
    return q and (q in text)

def looks_like_echo_of_sentence(quote: str, sentence: str) -> bool:
    q = normalize_space(quote).lower().strip(' "')
    s = normalize_space(sentence).lower()
    # treat identical or near-identical as echo
    return q == s or (len(q) > 20 and q in s)

def enforce_evidence(verdict: MicroVerdict, sentence: str) -> MicroVerdict:
    if not verdict.quote:
        verdict.supports = False
        verdict.score = 0.0
        verdict.invalid_reason = "no_quote"
        return verdict
    if looks_like_echo_of_sentence(verdict.quote, sentence):
        verdict.supports = False
        verdict.score = 0.0
        verdict.invalid_reason = "quote_equals_sentence"
        return verdict
    if not evidence_in_text(verdict.quote, verdict.title, verdict.abstract):
        verdict.supports = False
        verdict.score = 0.0
        verdict.invalid_reason = "quote_not_in_abstract"
        return verdict
    return verdict

# Optional: cue guards to tune function fit

CUES = {
    # Scene-setting, reviews, definitions, surveys
    "Background": re.compile(
        r"\b("
        r"we\s+(review|survey|summarize|describe|provide|present)\b"
        r"|overview|introduction|background|foundations?|primer"
        r"|define|definition|we\s+discuss\b"
        r")", re.I),

    # Applying a known method/tool/dataset to do something (implementation/use)
    "Uses": re.compile(
        r"\b("
        r"we\s+(use|apply|leverage|employ|utili[sz]e)\b"
        r"|implemented?|deploy|adopt|run|evaluate\s+using"
        r"|based\s+on\s+(?:the|a)\s+(method|model|approach)"
        r")", re.I),

    # Head-to-head comparisons, baselines, vs., outperform/underperform
    "Compares": re.compile(
        r"\b("
        r"compare[sd]?\b|comparison\b|versus|vs\."
        r"|against\b|relative\s+to\b|baseline[s]?\b"
        r"|outperform[s]?\b|superior\s+to|improv(es|ed)\s+over"
        r")", re.I),

    # Improves or generalizes a prior method (explicit extension language)
    "Extends": re.compile(
        r"\b("
        r"extend[s|ed|ing]?\b|generaliz(e|es|ed|ing)\b"
        r"|improv(e|es|ed|ing)\b\s+(?:on|over)\b"
        r"|build[s]?\s+on\b|augment[s]?\b|enhanc(e|es|ed|ing)\b"
        r"|refine[s|d|ment]\b|a\s+novel\s+(?:variant|extension)"
        r")", re.I),

    # Explicit follow-up/replication/continuation of a line of work
    "Continuation": re.compile(
        r"\b("
        r"follow-?up\b|we\s+revisit\b|continu(e|ation)\b"
        r"|replicat(e|ion)\b|extend\s+our\s+previous\s+work\b"
        r"|build\s+on\s+our\s+earlier\s+study\b"
        r")", re.I),

    # Identifies gaps/limits or proposes next steps
    "Motivation": re.compile(
        r"\b("
        r"limit(?:ation)?s?\b|bottleneck|problem|challenge|gap|lack\b"
        r"|uncertain|inadequate|insufficient|fail(?:s|ed|ure)"
        r"|breaks?\s*down|hard\s*to|difficult\s+to|cannot|open\s+issue"
        r")", re.I),

    # Forward-looking cues & proposals
    "FutureWork": re.compile(
        r"\b("
        r"future\s+(work|directions?|prospects?)"
        r"|we\s+(?:will|plan|aim|intend)\b"
        r"|remain(?:s|ed)?\s+to\s+(?:be\s+)?(done|explored|investigate)"
        r"|should\s+(?:be\s+)?(explored|investigated|addressed)"
        r"|further\s+(work|investigation|research)"
        r"|next\s+steps?|roadmap|open\s+question"
        r"|future\s+bounds?|further\s+testing"
        r")", re.I),
}
def enforce_function_cue(verdict: MicroVerdict, func: str) -> MicroVerdict:
    pat = CUES.get(func)
    if not pat:
        return verdict
    if verdict.supports and verdict.quote and not pat.search(verdict.quote or ""):
        verdict.score = min(verdict.score, 0.5)
    return verdict

# -------------------------
# Verify one candidate (with one retry)
# -------------------------
def verify_candidate(sentence: str, func: str, cand: Dict[str, Any], allow_retries: bool = True) -> MicroVerdict:
    func_norm = normalize_function(func)
    base_prompt = MICRO_PROMPT.format(
        SENTENCE=cand_safe(sentence, 2500),
        FUNCTION=func_norm,
        TITLE=cand_safe(cand.get("title",""), 900),
        ABSTRACT=cand_safe(cand.get("abstract",""), 2500),
        PAPER_ID=cand.get("paper_id","")
    )
    raw = llm_chat(base_prompt, model=DEFAULT_MODEL, temperature=0.0, max_tokens=900)
    obj = parse_json_strict(raw)

    v = MicroVerdict(
        supports=bool(obj.get("supports", False)),
        score=clamp01(obj.get("score", 0)),
        topicality=clamp01(obj.get("topicality", 0)),
        quote=str(obj.get("quote", ""))[:500],
        why=str(obj.get("why", ""))[:300],
        paper_id=str(obj.get("paper_id", cand.get("paper_id",""))),
        id=int(cand.get("rank", cand.get("id", 0)) or 0),
        title=cand.get("title", ""),
        abstract=cand.get("abstract", ""),
        authors=cand.get("authors", ""),
        year=str(cand.get("year", "")),
        retrieval_score=float(cand.get("retrieval_score") or 0.0)
    )
    v = enforce_evidence(v, sentence)
    v = enforce_function_cue(v, func_norm)

    # Retry only if the quote failed constraints
    if allow_retries and (v.invalid_reason in {"quote_equals_sentence", "quote_not_in_abstract", "no_quote"}):
        retry_prefix = base_prompt + \
            f"\n\nYour previous quote was invalid because: {v.invalid_reason}.\n" \
            "Pick a NEW quote that is an exact substring of TITLE or ABSTRACT only (<= 40 words).\n" \
            "Do NOT copy or paraphrase the SENTENCE. Return strict JSON."
        raw2 = llm_chat(retry_prefix, model=DEFAULT_MODEL, temperature=0.0, max_tokens=900)
        obj2 = parse_json_strict(raw2)
        v2 = MicroVerdict(
            supports=bool(obj2.get("supports", False)),
            score=clamp01(obj2.get("score", 0)),
            topicality=clamp01(obj2.get("topicality", 0)),
            quote=str(obj2.get("quote", ""))[:500],
            why=str(obj2.get("why", ""))[:300],
            paper_id=str(obj2.get("paper_id", v.paper_id)),
            id=v.id,
            title=v.title,
            abstract=v.abstract,
            authors=v.authors,
            year=v.year,
            retrieval_score=v.retrieval_score
        )
        v2 = enforce_evidence(v2, sentence)
        v2 = enforce_function_cue(v2, func_norm)
        if not v2.invalid_reason and (v2.supports or not v.supports):
            return v2
    return v

# -------------------------
# Aggregation
# -------------------------
def aggregate_verdicts(verdicts: List[MicroVerdict],
                       keep_top: int = 3,
                       tau_support: float = 0.55,
                       topical_floor: float = 0.60) -> Dict[str, Any]:
    supported = [v for v in verdicts if v.supports and v.score >= tau_support and not v.invalid_reason]
    supported.sort(key=lambda v: (v.score, v.topicality, v.retrieval_score), reverse=True)
    winners = supported[:keep_top]

    if not winners:
        valid_but_weak = [v for v in verdicts if not v.invalid_reason and v.quote and v.topicality >= topical_floor]
        valid_but_weak.sort(key=lambda v: (v.retrieval_score, v.topicality, v.score), reverse=True)
        winners = valid_but_weak[:min(2, keep_top)]

    out_candidates = []
    for v in verdicts:
        out_candidates.append({
            "id": v.id,
            "paper_id": v.paper_id,
            "title": v.title,
            "fit": "Fit" if v in winners else "NoFit",
            "scores": {
                "retrieval": round(v.retrieval_score, 3),
                "topicality": round(v.topicality, 3),
                "function_fit": round(v.score, 3),
                "confidence": round(0.4*v.score + 0.4*v.topicality + 0.2*v.retrieval_score, 3)
            },
            "rationale": v.why or ("invalid: " + (v.invalid_reason or "—")),
            "invalid_reason": v.invalid_reason or "",
            "evidence_spans": ([{"quote": v.quote, "why": v.why}] if v.quote else []),
            "authors": v.authors,
            "year": v.year
        })

    top_k = [v.id for v in winners]
    return {"candidates": out_candidates, "top_k": top_k}

# -------------------------
# Load & group
# -------------------------
def load_topk(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows

def group_by_sentence(rows: List[Dict[str, Any]]):
    grouped = defaultdict(list)
    meta = {}
    for r in rows:
        sid = r.get("sentence_uid") or f"sid{r.get('sentence_idx', 0)}"
        grouped[sid].append(r)
        func = r.get("citation_function")
        if not func:
            cf_list = r.get("citation_functions")
            if isinstance(cf_list, list) and cf_list:
                func = cf_list[0]
        meta[sid] = {
            "sentence": r.get("sentence_text", ""),
            "function": func or "Background"
        }
    for sid, lst in grouped.items():
        lst.sort(key=lambda x: (x.get("rank") is None, x.get("rank", 1e9)))
    return grouped, meta

def first_author(authors_str: str) -> str:
    if not authors_str:
        return ""
    parts = [a.strip() for a in re.split(r"[;,]| and ", authors_str) if a.strip()]
    if not parts:
        return authors_str.strip()
    cand = parts[0]
    tokens = cand.split()
    return tokens[-1] if tokens else cand

# -------------------------
# Pretty print block
# -------------------------
def print_block_header(sid, func, sentence):
    print("\n" + "="*80)
    print(f"Sentence UID: {sid}")
    print(f"Function    : {func}")
    print(f"Sentence    : {sentence}")
    print("-"*80)

def print_verdict(v: MicroVerdict, winner_ids: set):
    star = "✅" if v.id in winner_ids else "  "
    print(f"{star} id={v.id:<3} paper_id={v.paper_id:<20} "
          f"fit={'Fit' if v.id in winner_ids else 'NoFit':<5} "
          f"score={v.score:.2f} topicality={v.topicality:.2f} invalid={v.invalid_reason or '—'}")
    abs_show = (v.abstract or "")[:800].replace("\n", " ").strip()
    print(f"    title   : {v.title}")
    print(f"    abstract: {abs_show if abs_show else '[no abstract]'}")
    if v.quote:
        print(f"    evidence: {v.quote}")
    if v.why:
        print(f"    why     : {v.why}")
    print()

# -------------------------
# Main
# -------------------------
def main():
    TOPK_PATH = "/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/outputs/topk_candidates.jsonl"
    OUT_DIR = "/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/outputs"
    PRETTY_PATH = os.path.join(OUT_DIR, "verification_results_pretty.json")
    JSONL_PATH  = os.path.join(OUT_DIR, "verification_results.jsonl")
    CITE_JSONL = os.path.join(OUT_DIR, "function_citations.jsonl")
    CITE_PRETTY = os.path.join(OUT_DIR, "function_citations_pretty.json")

    KEEP_TOP = 3
    TAU_SUPPORT = 0.55
    TOPICAL_FLOOR = 0.60
    MAX_CHECK_PER_SENT = 10

    os.makedirs(OUT_DIR, exist_ok=True)

    rows = load_topk(TOPK_PATH)
    if not rows:
        print(f"No rows found in {TOPK_PATH}")
        return

    grouped, meta = group_by_sentence(rows)

    pretty_bundle = []
    cite_bundle = []

    for sid, cands in grouped.items():
        sentence = meta[sid]["sentence"]
        func = meta[sid]["function"]
        func_norm = normalize_function(func)

        print_block_header(sid, func_norm, sentence)

        verdicts: List[MicroVerdict] = []
        for c in cands[:MAX_CHECK_PER_SENT]:
            v = verify_candidate(sentence, func_norm, {
                "paper_id": c.get("paper_id") or c.get("arxiv_id") or "",
                "title": c.get("title", ""),
                "abstract": c.get("abstract", ""),
                "rank": c.get("rank", 0),
                "retrieval_score": c.get("normalized_score") or c.get("cosine_norm") or c.get("cosine") or 0.0,
                "authors": c.get("authors", ""),
                "year": c.get("year", "")
            }, allow_retries=True)
            verdicts.append(v)

        agg = aggregate_verdicts(verdicts, keep_top=KEEP_TOP,
                                 tau_support=TAU_SUPPORT, topical_floor=TOPICAL_FLOOR)

        winners = set(agg["top_k"])
        verdicts_sorted = sorted(
            verdicts,
            key=lambda v: ((v.id in winners), v.supports, v.score, v.topicality, v.retrieval_score),
            reverse=True
        )

        for v in verdicts_sorted:
            print_verdict(v, winners)

        num_valid = sum(1 for x in verdicts if not x.invalid_reason and x.quote)
        num_eqsent = sum(1 for x in verdicts if x.invalid_reason == "quote_equals_sentence")
        num_noquote = sum(1 for x in verdicts if x.invalid_reason == "no_quote")
        num_notin = sum(1 for x in verdicts if x.invalid_reason == "quote_not_in_abstract")
        print(f"Summary: valid_quotes={num_valid} | quote_equals_sentence={num_eqsent} "
              f"| no_quote={num_noquote} | quote_not_in_abstract={num_notin}")

        pretty_obj = {
            "sentence_uid": sid,
            "function": func_norm,
            "sentence_text": sentence,
            "top_k": agg["top_k"],
            "candidates": agg["candidates"]
        }
        pretty_bundle.append(pretty_obj)

        print("Aggregated:")
        print(json.dumps(pretty_obj, indent=2, ensure_ascii=False))

        # ---- Function-aware citation sentence for the best match ----
        top_verdict = None
        if agg["top_k"]:
            top_id = agg["top_k"][0]
            for v in verdicts_sorted:
                if v.id == top_id:
                    top_verdict = v
                    break
        if top_verdict is None:
            valid = [v for v in verdicts_sorted if not v.invalid_reason and v.quote]
            if valid:
                top_verdict = valid[0]

        if top_verdict:
            fa = first_author(top_verdict.authors)
            yr_match = re.findall(r"\d{4}", str(top_verdict.year) or "")
            yr = yr_match[0] if yr_match else ""
            cit_prompt = CITE_SENT_PROMPT.format(
                FUNCTION=func_norm,
                SENTENCE=sentence,
                TITLE=cand_safe(top_verdict.title, 600),
                ABSTRACT=cand_safe(top_verdict.abstract, 2500),
                FIRST_AUTHOR=fa,
                YEAR=yr
            )
            try:
                cite_sentence = llm_text(cit_prompt, model=DEFAULT_MODEL, temperature=0.2, max_tokens=120)
            except Exception:
                cite_sentence = ""

            cite_obj = {
                "sentence_uid": sid,
                "function": func_norm,
                "input_sentence": sentence,
                "paper_id": top_verdict.paper_id,
                "title": top_verdict.title,
                "authors": top_verdict.authors,
                "year": top_verdict.year,
                "evidence_quote": top_verdict.quote,
                "generated_citation_sentence": cite_sentence,
            }
            cite_bundle.append(cite_obj)

            print("\n--- Function-aware citation (top match) ---")
            print(json.dumps(cite_obj, indent=2, ensure_ascii=False))
        else:
            print("\n--- Function-aware citation (top match) ---")
            print("No suitable verified candidate to generate a citation sentence.")

    # SAVE verification outputs
    with open(PRETTY_PATH, "w", encoding="utf-8") as f:
        json.dump(pretty_bundle, f, indent=2, ensure_ascii=False)
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for obj in pretty_bundle:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # SAVE function-aware citation outputs
    with open(CITE_PRETTY, "w", encoding="utf-8") as f:
        json.dump(cite_bundle, f, indent=2, ensure_ascii=False)
    with open(CITE_JSONL, "w", encoding="utf-8") as f:
        for obj in cite_bundle:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"\n📄 Saved verification (pretty): {PRETTY_PATH}")
    print(f"📄 Saved verification (jsonl) : {JSONL_PATH}")
    print(f"📌 Saved function-citations (pretty): {CITE_PRETTY}")
    print(f"📌 Saved function-citations (jsonl)  : {CITE_JSONL}")

if __name__ == "__main__":
    main()
