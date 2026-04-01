#!/usr/bin/env python3
# verify_topk_strict_retry.py
# Verify function-fit of retrieval candidates with strict ABSTRACT-only evidence,
# retry once on invalid evidence, aggregate winners (excluding invalid),
# print details, and save pretty JSON + JSONL.

import os
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any

# -------------------------
# OpenAI-compatible client (your scads endpoint)
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
    """Use a system message to demand strict JSON and give the user content separately."""
    resp = CLIENT.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return STRICT JSON only. No prose, no markdown, no code fences."},
            {"role": "user", "content": prompt_user}
        ],
        temperature=temperature,
        max_tokens=max_tokens
        # If your server supports it, you can enforce JSON:
        # , response_format={"type":"json_object"}
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
    """
    Extract the first top-level JSON object by balancing braces.
    Works even if the model adds extra text before/after.
    """
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
# Micro-verifier prompt (ESCAPED BRACES!)
# -------------------------
MICRO_PROMPT = """You are a scientific function checker.

Normalize the requested function to one of:
Background, Uses, Compares, Motivation, Extends, FutureWork.

Function cues (synonyms allowed):
- Background: review/overview/introduce/describe/summarize/survey/tutorial
- Uses: use/apply/implement/employ/applied to/experiment using
- Compares: compare/comparison/versus/vs./relative to/benchmark/outperforms
- Motivation: limitation/challenge/problem/gap/uncertainty/bottleneck/need for/cannot explain/difficulty
- Extends: extend/extension/build upon/generalize/improve upon/adds to/broaden
- FutureWork: future work/plan/possible extensions/remains to be explored/further research/outlook/open question/directions

STRICT RULES:
- Your evidence MUST be a direct quote from the ABSTRACT only (<= 40 words).
- Do NOT paraphrase the SENTENCE. Do NOT copy from SENTENCE.
- If you cannot find function-specific evidence in the ABSTRACT, set "supports": false and "score": 0.0.

Return strict JSON:
{{
  "supports": true/false,
  "score": 0-1,
  "topicality": 0-1,
  "quote": "<<=40 words from ABSTRACT>>",
  "why": "short reason tied to the function cue",
  "paper_id": "{PAPER_ID}"
}}

SENTENCE: {SENTENCE}
FUNCTION: {FUNCTION}
TITLE: {TITLE}
ABSTRACT: {ABSTRACT}
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
    invalid_reason: str = ""      # set if our post-check rejects the evidence
    retrieval_score: float = 0.0  # retrieval similarity from your pipeline

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
# Evidence validation (anti-hallucination)
# -------------------------
def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def evidence_in_text(quote: str, title: str, abstract: str) -> bool:
    if not quote:
        return False
    # exact substring, case-insensitive
    q = normalize_space(quote).lower().strip(' "')
    text = normalize_space((title or "") + " " + (abstract or "")).lower()
    return q and (q in text)

def looks_like_echo_of_sentence(quote: str, sentence: str) -> bool:
    q = normalize_space(quote).lower().strip(' "')
    s = normalize_space(sentence).lower()
    return q == s or (len(q) > 0 and len(s) > 0 and (q in s or s in q))

def enforce_evidence(verdict: MicroVerdict, sentence: str) -> MicroVerdict:
    """If the quote isn't literally inside TITLE/ABSTRACT (or it just echoes the sentence), invalidate."""
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

# -------------------------
# Verify one candidate (with one retry on invalid evidence)
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
        id=int(cand.get("rank", cand.get("id", 0))),  # prefer retrieval rank; fallback to candidate id
        title=cand.get("title", ""),
        abstract=cand.get("abstract", ""),
        retrieval_score=float(cand.get("retrieval_score") or 0.0)
    )
    v = enforce_evidence(v, sentence)

    if allow_retries and (v.invalid_reason in {"quote_equals_sentence", "quote_not_in_abstract", "no_quote"}):
        retry_prompt = f"""{MICRO_PROMPT}

Your previous quote was invalid because: {v.invalid_reason}.
Pick a NEW quote that is an exact substring of ABSTRACT only (<= 40 words).
Do NOT copy or paraphrase the SENTENCE. Return strict JSON.
""".format(
            SENTENCE=cand_safe(sentence, 2500),
            FUNCTION=func_norm,
            TITLE=cand_safe(cand.get("title",""), 900),
            ABSTRACT=cand_safe(cand.get("abstract",""), 2500),
            PAPER_ID=cand.get("paper_id","")
        )
        raw2 = llm_chat(retry_prompt, model=DEFAULT_MODEL, temperature=0.0, max_tokens=900)
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
            retrieval_score=v.retrieval_score
        )
        v2 = enforce_evidence(v2, sentence)
        # Prefer the retry if it becomes valid or is better supported
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
    # Primary winners: must be supported, above threshold, and valid
    supported = [v for v in verdicts if v.supports and v.score >= tau_support and not v.invalid_reason]
    supported.sort(key=lambda v: (v.score, v.topicality, v.retrieval_score), reverse=True)
    winners = supported[:keep_top]

    if not winners:
        # Fallback MUST use valid quotes only (no invalid_reason), and with some topicality.
        valid_but_weak = [
            v for v in verdicts
            if not v.invalid_reason and v.quote and v.topicality >= topical_floor
        ]
        # Prefer retrieval score first, then topicality, then LLM score.
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
            "evidence_spans": ([{"quote": v.quote, "why": v.why}] if v.quote else [])
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
    # sort candidates by rank if present
    for sid, lst in grouped.items():
        lst.sort(key=lambda x: (x.get("rank") is None, x.get("rank", 1e9)))
    return grouped, meta

# -------------------------
# Main
# -------------------------
def main():
    TOPK_PATH = "/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/outputs/topk_candidates.jsonl"   # adjust if needed
    OUT_DIR = "outputs"
    PRETTY_PATH = os.path.join(OUT_DIR, "verification_results_pretty.json")
    JSONL_PATH  = os.path.join(OUT_DIR, "verification_results.jsonl")

    KEEP_TOP = 3
    TAU_SUPPORT = 0.55
    TOPICAL_FLOOR = 0.60
    MAX_CHECK_PER_SENT = 10   # verify up to top-10 per sentence

    os.makedirs(OUT_DIR, exist_ok=True)

    rows = load_topk(TOPK_PATH)
    if not rows:
        print("No rows found in outputs/topk_candidates.jsonl")
        return

    grouped, meta = group_by_sentence(rows)

    pretty_bundle = []

    for sid, cands in grouped.items():
        sentence = meta[sid]["sentence"]
        func = meta[sid]["function"]
        func_norm = normalize_function(func)

        print("\n" + "="*80)
        print(f"Sentence UID: {sid}")
        print(f"Function    : {func_norm}")
        print(f"Sentence    : {sentence}")
        print("-"*80)

        verdicts: List[MicroVerdict] = []
        for c in cands[:MAX_CHECK_PER_SENT]:
            v = verify_candidate(sentence, func_norm, {
                "paper_id": c.get("paper_id") or c.get("arxiv_id") or "",
                "title": c.get("title", ""),
                "abstract": c.get("abstract", ""),
                "rank": c.get("rank", 0),
                "retrieval_score": c.get("normalized_score") or c.get("cosine_norm") or c.get("cosine")
            }, allow_retries=True)
            verdicts.append(v)

        agg = aggregate_verdicts(verdicts, keep_top=KEEP_TOP,
                                 tau_support=TAU_SUPPORT, topical_floor=TOPICAL_FLOOR)

        # PRINT per-candidate verdicts (winners first), with TITLE + ABSTRACT
        winners = set(agg["top_k"])
        verdicts_sorted = sorted(
            verdicts,
            key=lambda v: ((v.id in winners), v.supports, v.score, v.topicality, v.retrieval_score),
            reverse=True
        )

        for v in verdicts_sorted:
            star = "✅" if v.id in winners else "  "
            print(f"{star} id={v.id:<3} paper_id={v.paper_id:<20} "
                  f"fit={'Fit' if v.id in winners else 'NoFit':<5} "
                  f"score={v.score:.2f} topicality={v.topicality:.2f} invalid={v.invalid_reason or '—'}")
            abs_show = (v.abstract or "")[:800].replace("\n", " ").strip()
            print(f"    title   : {v.title}")
            print(f"    abstract: {abs_show if abs_show else '[no abstract]'}")
            if v.quote:
                print(f"    evidence: {v.quote}")
            if v.why:
                print(f"    why     : {v.why}")
            print()

        # Per-sentence summary
        num_valid = sum(1 for x in verdicts if not x.invalid_reason and x.quote)
        num_eqsent = sum(1 for x in verdicts if x.invalid_reason == "quote_equals_sentence")
        num_noquote = sum(1 for x in verdicts if x.invalid_reason == "no_quote")
        num_notin = sum(1 for x in verdicts if x.invalid_reason == "quote_not_in_abstract")
        print(f"Summary: valid_quotes={num_valid} | quote_equals_sentence={num_eqsent} "
              f"| no_quote={num_noquote} | quote_not_in_abstract={num_notin}")

        # Build pretty JSON object for this sentence
        pretty_obj = {
            "sentence_uid": sid,
            "function": func_norm,
            "sentence_text": sentence,
            "top_k": agg["top_k"],
            "candidates": agg["candidates"]
        }
        pretty_bundle.append(pretty_obj)

        # Also print aggregated JSON for quick view
        print("Aggregated:")
        print(json.dumps(pretty_obj, indent=2, ensure_ascii=False))

    # SAVE pretty JSON (single multi-line JSON array)
    with open(PRETTY_PATH, "w", encoding="utf-8") as f:
        json.dump(pretty_bundle, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Saved pretty JSON: {PRETTY_PATH}")

    # SAVE JSONL (one object per line)
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for obj in pretty_bundle:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"📄 Saved JSONL:       {JSONL_PATH}")

if __name__ == "__main__":
    main()
