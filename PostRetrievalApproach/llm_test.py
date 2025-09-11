# llm_test.py
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

# Choose a model
models = client.models.list().data
preferred_model = next((m.id for m in models if "llama" in m.id.lower()), models[0].id)
print(f"✅ Using model: {preferred_model}")

# ---------- Few-shot examples ----------
few_shot_path = "few_shot_examples.txt"
if os.path.exists(few_shot_path):
    with open(few_shot_path, "r", encoding="utf-8") as f:
        few_shot_examples = f.read().strip()
else:
    print("⚠️ few_shot_examples.txt not found. Using minimal defaults.")
    few_shot_examples = """
You are a scientific assistant that decides whether a sentence needs a citation and what type it is.
Label set: Background, Uses, Extends, Compares, Motivation, FutureWork

Example A
Sentence: "Convolutional neural networks have achieved state-of-the-art performance in image classification."
needs_citation: true
citation_function: Background
justification: Introduces established field knowledge.

Example B
Sentence: "We use the ResNet50 architecture for feature extraction."
needs_citation: true
citation_function: Uses
justification: Describes use of a known method.

Example C
Sentence: "Our approach extends BERT by adding a domain-adaptive pretraining phase."
needs_citation: true
citation_function: Extends
justification: Builds on a prior method.

Example D
Sentence: "Compared to HMMER, our model recovers more remote homologs at fixed FPR."
needs_citation: true
citation_function: Compares
justification: Makes a comparative performance claim.

Example E
Sentence: "Identifying remote homologs with low sequence identity remains challenging."
needs_citation: true
citation_function: Motivation
justification: States a problem/gap motivating the work.

Example F
Sentence: "In future work, we will incorporate structural templates."
needs_citation: false
citation_function: FutureWork
justification: Outlines planned work; no citation needed.
""".strip()

# ---------- Input ----------
user_query = input("\n🔍 Enter a scientific query or topic: ").strip()

# ---------- Generation prompt (no references, no lists) ----------
topic_prompt = f"""
You are a scientific writing assistant.

Write a single coherent paragraph answering the following scientific question:
{user_query}

Requirements:
- The paragraph must be at least 6 sentences long.
- It should contain a mix of factual background, use of methods, comparisons, extensions of prior work, motivations, and possible future work.
- Each sentence should present one scientific claim that could be cited in the future.
- When referencing prior work, include author-year style references in parentheses, e.g., (Smith et al., 2020).
- Do NOT use bullet points, numbered lists, subheadings, or include a references section.
- The tone should be formal and suitable for a peer-reviewed scientific paper.
- Avoid actual citation placeholders like “[1]” or full bibliographic details.
"""


topic_resp = client.chat.completions.create(
    model=preferred_model,
    messages=[{"role": "user", "content": topic_prompt}],
    temperature=0.5,
    max_tokens=512
)
raw_text = topic_resp.choices[0].message.content.strip()

# ---------- Sanitize any sneaky refs/lists ----------
def clean_text(text: str) -> str:
    # Remove anything after a References: section
    text = re.split(r'\bReferences\s*:?\s*$', text, flags=re.IGNORECASE | re.MULTILINE)[0]
    # Strip standalone citation blocks like [1], (1)
    text = re.sub(r'\s*\[\d+\]\s*', ' ', text)
    text = re.sub(r'\s*\(\d+\)\s*', ' ', text)
    # Remove bullet/list markers at line starts
    text = re.sub(r'^\s*(?:[-*•]|\d+\.)\s+', '', text, flags=re.MULTILINE)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

clean_paragraph = clean_text(raw_text)

print("\n📝 Generated Paragraph (sanitized):\n")
print(clean_paragraph)

# ---------- Split into sentences ----------
sentences = re.split(r'(?<=[.!?])\s+', clean_paragraph)
sentences = [s.strip() for s in sentences if s.strip()]

# ---------- Helper: robust JSON extractor ----------
def extract_json_only(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"error": "No JSON block found", "raw": text}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"error": "JSON parse failed", "raw": text}

# ---------- Classify each sentence ----------
results = []
for i, sent in enumerate(sentences, 1):
    # Skip garbage/non-sentences
    if len(sent) < 4 or re.fullmatch(r'[\[\]().,:;0-9 -]+', sent):
        continue

    print(f"\n🔍 Sentence {i}: {sent}\n")

    classification_prompt = f"""
{few_shot_examples}

Classify the sentence below. Return ONLY a single JSON object and nothing else.

Sentence: "{sent}"

JSON schema:
{{
  "needs_citation": true/false,
  "citation_function": "Background|Uses|Extends|Compares|Motivation|FutureWork",
  "justification": "short"
}}
""".strip()

    cls = client.chat.completions.create(
        model=preferred_model,
        messages=[{"role": "user", "content": classification_prompt}],
        temperature=0.0,
        max_tokens=200
    )
    parsed = extract_json_only(cls.choices[0].message.content.strip())
    print("\n📅 Parsed Result:\n", json.dumps(parsed, indent=2))
    results.append({"sentence": sent, **parsed})

# ---------- (Optional) save to JSONL ----------
out = {
    "query": user_query,
    "model": preferred_model,
    "paragraph": clean_paragraph,
    "sentence_classification": results,
}

OVERWRITE = True  # set False if you want to keep history

mode = "w" if OVERWRITE else "a"
with open("classified_outputs.jsonl", mode, encoding="utf-8") as f:
    f.write(json.dumps(out) + "\n")

print("\n✅ Appended to classified_outputs.jsonl")
