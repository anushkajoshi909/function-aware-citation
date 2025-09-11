import os
import json
import re
from openai import OpenAI

# ---------- Setup ----------
api_key_path = os.path.join(os.path.expanduser("~"), ".scadsai-api-key")
if not os.path.exists(api_key_path):
    print("❌ API key file not found at ~/.scadsai-api-key")
    exit(1)
with open(api_key_path) as f:
    api_key = f.read().strip()
client = OpenAI(base_url="https://llm.scads.ai/v1", api_key=api_key)

# ---------- Choose Model ----------
models = client.models.list().data
preferred_model = next((m.id for m in models if "llama" in m.id.lower()), models[0].id)
print(f"✅ Using model: {preferred_model}")

# ---------- Few-Shot Examples ----------
few_shot_examples = """
You are a scientific assistant that classifies the purpose of a scientific query.

Label set: Background, Uses, Extends, Compares, Motivation, FutureWork

Return a list of 1–2 most relevant citation functions that apply to the query.
Also provide a short justification.

---

Example 1:
Query: "What is the Transformer architecture?"
{
  "citation_functions": ["Background"],
  "justification": "The query seeks foundational information."
}

Example 2:
Query: "How is BERT used in clinical NLP tasks?"
{
  "citation_functions": ["Uses"],
  "justification": "The query focuses on practical applications of BERT."
}

Example 3:
Query: "What background about the QCD phase transition motivates its investigation for nucleosynthesis?"
{
  "citation_functions": ["Background", "Motivation"],
  "justification": "The query introduces foundational concepts and highlights a reason for further study."
}

Example 4:
Query: "How does BioBERT improve on BERT for biomedical tasks?"
{
  "citation_functions": ["Extends", "Uses"],
  "justification": "The query focuses on an extension and its usage in a domain."
}

Example 5:
Query: "What challenges in multi-label classification motivate hierarchical learning methods?"
{
  "citation_functions": ["Motivation"],
  "justification": "The query identifies a problem area that motivates research."
}

Example 6:
Query: "How do CNNs and Vision Transformers compare on ImageNet?"
{
  "citation_functions": ["Compares", "Uses"],
  "justification": "The query makes a comparative and usage-based inquiry."
}
""".strip()

def extract_json_only(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"error": "No JSON block found", "raw": text}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"error": "JSON parse failed", "raw": text}

def clean_text(text: str) -> str:
    # Remove any trailing "References" sections or bracketed numeric refs
    text = re.split(r'\bReferences\s*:?\s*$', text, flags=re.IGNORECASE | re.MULTILINE)[0]
    text = re.sub(r'\s*\[\d+\]\s*', ' ', text)
    text = re.sub(r'\s*\(\d+\)\s*', ' ', text)
    text = re.sub(r'^\s*(?:[-*•]|\d+\.)\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------- Input Query ----------
user_query = input("\n🔍 Enter a scientific query or topic: ").strip()

# ---------- Step 1: Classify Citation Functions ----------
classification_prompt = f"""
{few_shot_examples}

Now classify this query. Return ONLY a single JSON object.

Query: "{user_query}"

JSON schema:
{{
  "citation_functions": ["Background", "Motivation"],  // up to 2
  "justification": "short explanation"
}}
""".strip()

cls_resp = client.chat.completions.create(
    model=preferred_model,
    messages=[{"role": "user", "content": classification_prompt}],
    temperature=0.0,
    max_tokens=300
)

classification_result = extract_json_only(cls_resp.choices[0].message.content.strip())
print("\n🔎 Citation Function Classification:")
print(json.dumps(classification_result, indent=2))

# Extract the ordered list of functions
citation_types = classification_result.get("citation_functions", [])
citation_types = [c for c in citation_types if isinstance(c, str) and c.strip()]
if not citation_types:
    # Fallback: if classifier failed, default to Background to keep pipeline running
    citation_types = ["Background"]
citation_str = ", ".join(citation_types)

# ---------- Step 2: Generate One Sentence Per Function (exact order & count) ----------
gen_n = len(citation_types)
topic_prompt = f"""
You are a scientific writing assistant.

Write exactly {gen_n} sentences — one per citation function — in response to this scientific query:

Query: "{user_query}"

Citation functions (in order): {citation_str}

Instructions:
- Write exactly one sentence per citation function, in the same order as listed.
- Do NOT add any extra sentences, headings, list markers, or explanations.
- Each sentence must serve only its respective citation function.
- Keep each sentence self-contained and citable.
- Formal tone suitable for peer-reviewed academic writing.
- Output a single paragraph composed only of these {gen_n} sentences.
""".strip()

topic_resp = client.chat.completions.create(
    model=preferred_model,
    messages=[{"role": "user", "content": topic_prompt}],
    temperature=0.5,
    max_tokens=512
)
raw_text = topic_resp.choices[0].message.content.strip()
clean_paragraph = clean_text(raw_text)

print("\n📝 Generated Paragraph:\n")
print(clean_paragraph)

# ---------- Split Into Sentences ----------
sentences = re.split(r'(?<=[.!?])\s+', clean_paragraph)
sentences = [s.strip() for s in sentences if s.strip()]

# Enforce exact count alignment with classification functions
if len(sentences) > gen_n:
    print(f"\n⚠️ Note: Model produced {len(sentences)} sentences but {gen_n} were requested; truncating extras.")
    sentences = sentences[:gen_n]
elif len(sentences) < gen_n:
    print(f"\n⚠️ Warning: Model produced only {len(sentences)} of {gen_n} requested sentences. "
          f"The remaining functions will be omitted from sentence-level output.")

# ---------- Attach Known Function Per Sentence (no re-classification) ----------
results = []
for i, sent in enumerate(sentences, 1):
    print(f"\n🔍 Sentence {i}: {sent}\n")
    func = citation_types[i - 1] if i - 1 < len(citation_types) else "Background"
    parsed = {
        "needs_citation": True,
        "citation_functions": [func],
        "justification": f"This sentence was generated to fulfill the '{func}' citation function."
    }
    print("\n📅 Parsed Result:\n", json.dumps(parsed, indent=2))
    results.append({"sentence": sent, **parsed})

# ---------- Save Output ----------
out = {
    "query": user_query,
    "citation_function_classification": classification_result,
    "model": preferred_model,
    "paragraph": clean_paragraph,
    "sentence_classification": results,
}

OVERWRITE = True  # set False if you want to keep history

mode = "w" if OVERWRITE else "a"
with open("classified_outputs.jsonl", mode, encoding="utf-8") as f:
    f.write(json.dumps(out) + "\n")

print("\n✅ Saved to classified_outputs.jsonl")
