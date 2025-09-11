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

# ---------- List available models ----------
models = client.models.list().data
print("✅ Available Models:")
for i, m in enumerate(models, 1):
    print(f"{i}. {m.id}")

# ---------- Choose model interactively ----------
choice = input("\n👉 Select a model number (default = 1): ").strip()
if choice.isdigit() and 1 <= int(choice) <= len(models):
    preferred_model = models[int(choice) - 1].id
else:
    preferred_model = models[0].id

print(f"\n✅ Using model: {preferred_model}")


# ---------- Few-Shot Examples (for classification only) ----------
# ---------- Few-Shot Examples (only multi-function ones) ----------
# ---------- Few-Shot Examples (only multi-function ones) ----------
few_shot_examples = """
You are a scientific assistant that classifies the purpose of a scientific query.

Label set: Background, Uses, Extends, Compares, Motivation, FutureWork

Return a list of 2 most relevant citation functions that apply to the query.
Also provide a short justification.

---

Example 1:
Query: "What background about the QCD phase transition motivates its investigation for nucleosynthesis?"
{
  "citation_functions": ["Background", "Motivation"],
  "justification": "The query introduces foundational concepts and highlights a reason for further study."
}

Example 2:
Query: "How does BioBERT improve on BERT for biomedical tasks?"
{
  "citation_functions": ["Extends", "Uses"],
  "justification": "The query focuses on an extension and its usage in a domain."
}

Example 3:
Query: "How do CNNs and Vision Transformers compare on ImageNet?"
{
  "citation_functions": ["Compares", "Uses"],
  "justification": "The query makes a comparative and usage-based inquiry."
}

Example 4:
Query: "What background on generative adversarial networks explains their use in image synthesis?"
{
  "citation_functions": ["Background", "Uses"],
  "justification": "The query requests foundational knowledge and also asks about practical application."
}

Example 5:
Query: "How do reinforcement learning methods extend supervised learning for robotics?"
{
  "citation_functions": ["Extends", "Uses"],
  "justification": "The query emphasizes how one approach builds on another and its practical usage in robotics."
}

Example 6:
Query: "What motivates the use of graph neural networks for social network analysis?"
{
  "citation_functions": ["Motivation", "Uses"],
  "justification": "The query highlights drivers behind the approach and asks about its application domain."
}

Example 7:
Query: "How does recent work on attention mechanisms compare with recurrent networks, and how are they applied in translation tasks?"
{
  "citation_functions": ["Compares", "Uses"],
  "justification": "The query is both comparative and focuses on usage in a specific application."
}

Example 8:
Query: "What background about protein folding motivates the use of deep learning in structural biology?"
{
  "citation_functions": ["Background", "Motivation"],
  "justification": "The query introduces a fundamental scientific concept and relates it to why new methods are explored."
}
Example 9:
Query: "How can attention-based models be used to guide future research in low-resource machine translation?"
{
  "citation_functions": ["Uses", "FutureWork"],
  "justification": "The query refers to how a method is applied and explicitly frames it as guidance for subsequent research."
}

Example 10:
Query: "What challenges in protein structure prediction motivate future work using deep generative models?"
{
  "citation_functions": ["Motivation", "FutureWork"],
  "justification": "The query highlights a motivating challenge and emphasizes directions for future research."
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

# ---------- Input Query ----------
user_query = input("\n🔍 Enter a scientific query or topic: ").strip()

# ---------- Step: Classify Citation Functions (no answer generation) ----------
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

# Fallback to keep downstream retrieval stable
if not isinstance(classification_result, dict) or "citation_functions" not in classification_result:
    classification_result = {
        "citation_functions": ["Background"],
        "justification": "Fallback: classifier did not return a valid object."
    }

print("\n🔎 Citation Function Classification:")
print(json.dumps(classification_result, indent=2))

# ---------- Save Minimal Output for Retrieval Stage ----------
out = {
    "query": user_query,
    "citation_function_classification": classification_result,
    "model": preferred_model,
    "note": "No answer/explanation generated. Use this record as input to the retrieval module."
}

OVERWRITE = True  # set False if you want to keep history
mode = "w" if OVERWRITE else "a"
with open("classified_outputs.jsonl", mode, encoding="utf-8") as f:
    f.write(json.dumps(out) + "\n")

print("\n✅ Saved to classified_outputs.jsonl")
print("🔧 Ready for retrieval module (handled in a separate file).")
