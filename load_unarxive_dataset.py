from datasets import load_dataset
import sys

# Enable real-time log output
sys.stdout.reconfigure(line_buffering=True)
print("🚀 Loading 1 source documents from unarxive_2024...", flush=True)

# Load just the first 10 documents (expanded into examples)
dataset = load_dataset("ines-besrour/unarxive_2024", split="train[:1]")

print(f"✅ Loaded {len(dataset)} expanded examples", flush=True)

# Save to disk
save_path = "/home/anpa439f/Research_Project/unarxive_2024_10docs"
dataset.save_to_disk(save_path)

print(f"💾 Saved dataset to: {save_path}", flush=True)
