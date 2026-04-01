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

OUTPUT_PATH = "output_dataset1.jsonl"  # Update to real path

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

random.seed(42)
papers = random.sample(all_papers, 200)

# -------- Helpers --------
def extract_json_from_response(response_text):
    if response_text.startswith("```"):
        response_text = response_text.strip().lstrip("`").rstrip("`")
        if response_text.startswith("json\n"):
            response_text = response_text[len("json\n"):]
    return response_text.strip()

def safe_json_load(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text_escaped = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
        return json.loads(text_escaped)

# -------- Generate dataset --------
def create_dataset():
    dataset = []
    print(f"✅ Generating data for {len(papers)} papers...\n", flush=True)

    for idx, paper in enumerate(papers):
        paper_id = paper.get("metadata", {}).get("id", f"UNKNOWN_ID_{idx}")
        abstract = paper.get("metadata", {}).get("abstract", "")

        print(f"📄 [{idx + 1}/{len(papers)}] Paper ID: {paper_id}", flush=True)

        if not abstract:
            print(f"⚠️ Skipping due to missing abstract.", flush=True)
            continue

        citation_functions = random.sample(list(CITATION_FUNCTION_DESCRIPTIONS.keys()), 2)

        citation_descriptions = "\n".join([
            f"- **{label}**: {desc}" for label, desc in CITATION_FUNCTION_DESCRIPTIONS.items()
        ])

        prompt = f"""
You are a scientific assistant.

Your task is to read the following abstract and:
- Generate **one meaningful question** that relates to **both** of the following citation functions:
  1. {citation_functions[0]}
  2. {citation_functions[1]}
- Provide an answer using only the content of the abstract.
- In the answer, mark each relevant sentence or clause with the matching citation function label in square brackets.

Citation Function Descriptions:
{citation_descriptions}

Abstract:
{abstract}

Return your response in the following JSON format:

{{
  "question": "<single question reflecting both citation functions>",
  "citation_functions": ["{citation_functions[0]}", "{citation_functions[1]}"],
  "answer": "<answer from abstract, annotated with [CitationFunction] tags>"
}}

Do not include any explanation. Return only the JSON object.
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

            dataset.append({
                "paper_id": paper_id,
                "question": output_json.get("question", ""),
                "citation_functions": output_json.get("citation_functions", []),
                "answer": output_json.get("answer", "")
            })

        except Exception as e:
            print(f"⚠️ Error generating for {paper_id}: {e}", flush=True)
            print(f"❓ Raw output:\n{result}\n", flush=True)

    # Write dataset
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Dataset saved to {OUTPUT_PATH}", flush=True)

# -------- Main --------
if __name__ == "__main__":
    create_dataset()
