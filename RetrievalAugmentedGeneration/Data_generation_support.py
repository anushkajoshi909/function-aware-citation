import os
import json
import random
import re
from openai import OpenAI

# ---------- SCADS client ----------
api_key_path = os.path.join(os.path.expanduser("~"), ".scadsai-api-key")
if not os.path.exists(api_key_path):
    print("❌ API key file not found at ~/.scadsai-api-key", flush=True)
    exit(1)
with open(api_key_path) as f:
    api_key = f.read().strip()
client = OpenAI(base_url="https://llm.scads.ai/v1", api_key=api_key)

# Pick model
models = client.models.list().data
PREFERRED_MODEL = next((m.id for m in models if "llama" in m.id.lower()), models[0].id)
print(f"✅ Using model: {PREFERRED_MODEL}", flush=True)

# Input JSONL files
jsonl_files = [
    "/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/Corpus/processed_unarxive_extended_data/unarXive_01981221/01/arXiv_src_0101_001.jsonl",
    "/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/Corpus/processed_unarxive_extended_data/unarXive_01981221/01/arXiv_src_0102_001.jsonl",
    "/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/Corpus/processed_unarxive_extended_data/unarXive_01981221/01/arXiv_src_0103_001.jsonl",
    "/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/Corpus/processed_unarxive_extended_data/unarXive_01981221/01/arXiv_src_0104_001.jsonl",
]

OUTPUT_PATH = "/data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/RetrievalAugmentedGeneration/outputs/synthetic_dataset_strict2.jsonl"  # Update to real path

# ---------- Abstract quality thresholds (multi-signal) ----------
MIN_ABS_CHARS = 600        # tighten/loosen as needed
MIN_ABS_WORDS = 150
MIN_ABS_SENTENCES = 5      # sentences with >= 6 words

LATEX_CMD_RE = re.compile(r'\\[a-zA-Z]+(\{.*?\})?')
MATH_INLINE_RE = re.compile(r'\$[^$]+\$')
MATH_DISPLAY_RE = re.compile(r'\$\$[^$]+\$\$')

def clean_abstract(text: str) -> str:
    """Remove LaTeX/math and condense whitespace for robust counting."""
    if not text:
        return ""
    t = text
    t = MATH_DISPLAY_RE.sub(' ', t)
    t = MATH_INLINE_RE.sub(' ', t)
    t = LATEX_CMD_RE.sub(' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def count_sentences(t: str) -> int:
    """Count substantial sentences (>= 6 words)."""
    if not t:
        return 0
    candidates = re.split(r'(?<=[.!?])\s+', t)
    substantial = [s for s in candidates if len(s.split()) >= 6]
    return len(substantial)

def abstract_meets_threshold(raw_abstract: str):
    """Return (ok: bool, metrics: dict, reason: str|None)."""
    t = clean_abstract(raw_abstract)
    chars = len(t)
    words = len(t.split())
    sents = count_sentences(t)

    if chars < MIN_ABS_CHARS:
        return (False, {"chars": chars, "words": words, "sents": sents}, "chars_below_min")
    if words < MIN_ABS_WORDS:
        return (False, {"chars": chars, "words": words, "sents": sents}, "words_below_min")
    if sents < MIN_ABS_SENTENCES:
        return (False, {"chars": chars, "words": words, "sents": sents}, "sents_below_min")
    return (True, {"chars": chars, "words": words, "sents": sents}, None)

# Citation function descriptions
CITATION_FUNCTION_DESCRIPTIONS = {
    "Background": "Provides general background information, related work, or context needed to understand the paper.",
    "Uses": "Indicates that the current work uses a method, dataset, or result from the cited work.",
    "Compares": "Compares this work with the cited work in terms of approach, performance, or results.",
    "Motivation": "Motivates the current work by identifying a problem, gap, or limitation in the cited work.",
    "Extends": "Builds on or improves the cited work's method, theory, or results.",
    "FutureWork": "Suggests areas for future research or extensions, often building on the cited work."
}

# -------- Read files --------
def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"⚠️ Skipping malformed line in {path}", flush=True)
    return rows

all_papers = []
for file_path in jsonl_files:
    all_papers.extend(read_jsonl(file_path))

# ✅ Select exactly 200 papers with sufficiently strong abstracts
random.seed(42)
all_indices = list(range(len(all_papers)))
random.shuffle(all_indices)

TARGET_SIZE = 200

papers = []
for idx in all_indices:
    if len(papers) >= TARGET_SIZE:
        break
    candidate = all_papers[idx]
    abstract = candidate.get("metadata", {}).get("abstract", "")
    ok, metrics, reason = abstract_meets_threshold(abstract)
    if not ok:
        # Skip weak/short abstracts and keep sampling until we reach TARGET_SIZE
        continue
    papers.append(candidate)

print(f"✅ Selected {len(papers)} papers meeting thresholds "
      f"(chars≥{MIN_ABS_CHARS}, words≥{MIN_ABS_WORDS}, sents≥{MIN_ABS_SENTENCES}).")

if len(papers) < TARGET_SIZE:
    print(f"⚠️ Only {len(papers)} papers met the thresholds; consider relaxing MIN_* values if you require exactly {TARGET_SIZE}.", flush=True)

# -------- Helpers --------
def extract_json_from_response(response_text):
    # unwrap code fences if any (we told it not to, but be safe)
    if response_text.startswith("```"):
        response_text = response_text.strip().lstrip("`").rstrip("`")
        if response_text.startswith("json\n"):
            response_text = response_text[len("json\n"):]
    return response_text.strip()

def safe_json_load(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text_escaped = re.sub(r'\\(?![\"\\/bfnrtu])', r'\\\\', text)
        return json.loads(text_escaped)

# ---------- NEW: support/unsupported tagging helpers ----------
ALL_FUNCS = {"Background","Uses","Compares","Motivation","Extends","FutureWork"}
CANON = {
    "background":"Background","uses":"Uses","compares":"Compares",
    "motivation":"Motivation","extends":"Extends",
    "futurework":"FutureWork","future_work":"FutureWork","future-work":"FutureWork"
}
NEG_PAT = re.compile(
    r"(does\s+not\s+provide|no\s+information\s+relevant|doesn['’]t\s+discuss|"
    r"does\s+not\s+discuss|not\s+discussed|no\s+explicit|no\s+evidence\s+of|absent)",
    re.I
)

def canon_fn(s: str) -> str:
    s = (s or "").strip()
    return CANON.get(s.lower(), s)

def infer_support_per_function(answer_text: str):
    """
    Parse bracket-tagged sentences like: '... [Uses]'.
    Return dict: {Function: {'supported': True/False, 'sentence': '<text>'}}
    """
    out = {}
    # capture (sentence_before_tag, FunctionTag)
    for sent, fn in re.findall(r'([^\[]+)\[([A-Za-z_-]+)\]', answer_text or ""):
        fn_c = canon_fn(fn)
        if fn_c in ALL_FUNCS:
            unsupported = bool(NEG_PAT.search(sent or ""))
            out[fn_c] = {"supported": (not unsupported), "sentence": sent.strip()}
    return out
# ---------- END helpers ----------

# -------- Generate dataset --------
def create_dataset():
    dataset = []
    print(f"✅ Generating data for {len(papers)} papers...\n", flush=True)

    for idx, paper in enumerate(papers):
        paper_id = paper.get("metadata", {}).get("id", f"UNKNOWN_ID_{idx}")
        abstract = paper.get("metadata", {}).get("abstract", "")

        print(f"📄 [{idx + 1}/{len(papers)}] Paper ID: {paper_id}", flush=True)

        # Double-check thresholds (defensive)
        ok, metrics, reason = abstract_meets_threshold(abstract)
        if not ok:
            print(f"⚠️ Skipping (post-check) reason={reason} "
                  f"chars={metrics['chars']} words={metrics['words']} sents={metrics['sents']}  — {paper_id}", flush=True)
            continue

        citation_functions = random.sample(list(CITATION_FUNCTION_DESCRIPTIONS.keys()), 2)

        citation_descriptions = "\n".join([
            f"- **{label}**: {desc}" for label, desc in CITATION_FUNCTION_DESCRIPTIONS.items()
        ])

        # Prompt (unchanged methodology)
        prompt = f"""
You are a scientific assistant.

Your task is to read the following abstract and:
- Generate **one meaningful question** that relates to **both** of the following citation functions:
  1. {citation_functions[0]}
  2. {citation_functions[1]}
- Provide an answer using **only** the content of the abstract (no outside knowledge).
- Write **exactly one sentence per citation function**, in the same order as listed.
- If the abstract does **not** contain enough information to answer for a given function, write a sentence saying so, still tagged with that function (e.g., "The abstract does not provide information relevant to FutureWork [FutureWork].").
- In the answer, mark each sentence with the matching citation function label in square brackets, e.g., "... [Background]".
- After the answer, provide a brief explanation that states how the abstract supports each sentence (or why it couldn’t be supported).

Citation Function Descriptions:
{citation_descriptions}

Abstract:
{abstract}

Return your response in the following JSON format, and return ONLY this JSON object:

{{
  "question": "<single question reflecting both citation functions>",
  "citation_functions": ["{citation_functions[0]}", "{citation_functions[1]}"],
  "answer": "<answer from abstract, annotated with [CitationFunction] tags>",
  "explanation": "<why this answer follows from the abstract (one short sentence per function, in order)>"
}}
"""

        try:
            response = client.chat.completions.create(
                model=PREFERRED_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Return only the JSON object. Do not use markdown formatting or backticks."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
                max_tokens=600
            )

            result = response.choices[0].message.content.strip()
            cleaned_result = extract_json_from_response(result)
            output_json = safe_json_load(cleaned_result)

            # --- NEW: derive support flags per function from the model's answer
            fn_list = [canon_fn(x) for x in output_json.get("citation_functions", [])]
            support_map = infer_support_per_function(output_json.get("answer", ""))

            # ensure every requested function has a label, even if missing in text
            labels = []
            for fn in fn_list:
                entry = support_map.get(fn, {"supported": False, "sentence": ""})
                labels.append({"function": fn, "supported": bool(entry["supported"])})

            supported_functions = [x["function"] for x in labels if x["supported"]]
            unsupported_functions = [x["function"] for x in labels if not x["supported"]]
            # --- END NEW

            dataset.append({
                "paper_id": paper_id,
                "question": output_json.get("question", ""),
                "citation_functions": fn_list,
                "answer": output_json.get("answer", ""),
                "explanation": output_json.get("explanation", ""),
                # NEW fields for eval
                "labels": labels,  # [{"function":"Uses","supported":true}, ...]
                "supported_functions": supported_functions,
                "unsupported_functions": unsupported_functions
            })

        except Exception as e:
            raw = locals().get("result", "<no result captured>")
            print(f"⚠️ Error generating for {paper_id}: {e}", flush=True)
            print(f"❓ Raw output:\n{raw}\n", flush=True)

    # Write dataset
    with open(OUTPUT_PATH, 'w', encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Dataset saved to {OUTPUT_PATH}", flush=True)

# -------- Main --------
if __name__ == "__main__":
    create_dataset()
