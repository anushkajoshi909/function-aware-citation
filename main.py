#!/usr/bin/env python3

import os
import json
import re
from datetime import datetime
from openai import OpenAI

# Load API key
api_key_path = os.path.join(os.path.expanduser("~"), ".scadsai-api-key")
if not os.path.exists(api_key_path):
    print("❌ API key file not found at ~/.scadsai-api-key")
    exit(1)

with open(api_key_path) as f:
    api_key = f.read().strip()

if not api_key:
    print("❌ API key file is empty.")
    exit(1)

# Initialize client
client = OpenAI(base_url="https://llm.scads.ai/v1", api_key=api_key)

# List available models
models = client.models.list().data
print("✅ Available Models:")
for m in models:
    print(f" - {m.id}")

# Choose a preferred model
preferred_model = next((m.id for m in models if "llama" in m.id), models[0].id)

# Ask for a topic
query = input("\n🔍 Enter your scientific topic: ").strip()

# Step 1: Generate paragraph
prompt = (
    f"Write a scientific paragraph about the topic below. "
    f"Do not include citations yet, but write as if it would include them later.\n\n"
    f"Topic: {query}\n\nAnswer:"
)

response = client.chat.completions.create(
    model=preferred_model,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
    max_tokens=512
)

generated_paragraph = response.choices[0].message.content.strip()
print("\n📝 Generated Paragraph:\n")
print(generated_paragraph)

# Step 2: Analyze each sentence
sentences = re.split(r'(?<=[.!?]) +', generated_paragraph)
print("\n🔍 Analyzing Each Sentence for Citation Need and Function:\n")
analysis = []

for sentence in sentences:
    prompt_analysis = (
        f"Sentence: \"{sentence}\"\n\n"
        f"Does this sentence need a citation? If yes, what is the purpose?\n"
        f"Choose one: Background, Uses, Extends, Compares, Motivation, FutureWork.\n"
        f"Respond in two parts:\n"
        f"1. JSON object with keys: needs_citation, citation_function, justification\n"
        f"2. One sentence explanation after the JSON"
    )

    analysis_response = client.chat.completions.create(
        model=preferred_model,
        messages=[{"role": "user", "content": prompt_analysis}],
        temperature=0,
        max_tokens=256
    )

    raw_response = analysis_response.choices[0].message.content.strip()

    # 📦 Print raw response for inspection
    print(f"\n📦 Raw LLM response for sentence:\n{sentence}\n---\n{raw_response}\n")

    # Try to extract JSON block
    match = re.search(r'\{.*?\}', raw_response, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))

            # Fix incorrect key if present
            if "purpose" in parsed and "citation_function" not in parsed:
                parsed["citation_function"] = parsed.pop("purpose")

            # Validate required keys
            if not all(k in parsed for k in ["needs_citation", "citation_function", "justification"]):
                parsed = {"error": "Missing expected keys", "raw_response": raw_response}

        except json.JSONDecodeError:
            parsed = {"error": "Could not parse JSON", "raw_response": raw_response}
    else:
        parsed = {"error": "No JSON block found", "raw_response": raw_response}

    analysis.append({
        "sentence": sentence,
        "analysis": parsed,
        "raw_response": raw_response
    })

# Display final analysis
for item in analysis:
    print(f"🔸 Sentence: {item['sentence']}")
    print(f"   ➤ Analysis: {json.dumps(item['analysis'], indent=2)}\n")

# Step 3: Save everything
entry = {
    "timestamp": datetime.utcnow().isoformat(),
    "model": preferred_model,
    "query": query,
    "generated_paragraph": generated_paragraph,
    "sentence_analysis": analysis
}

with open("generated_responses.jsonl", "a") as f:
    f.write(json.dumps(entry) + "\n")

print("✅ All results saved to 'generated_responses.jsonl'")
