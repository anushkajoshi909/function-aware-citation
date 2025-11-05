# Function-Aware Citation Retrieval and Generation

## Overview
Large Language Models (LLMs) have transformed natural language processing, enabling powerful applications in question answering, summarization, and knowledge synthesis. Despite these advances, a persistent challenge remains: ensuring that generated outputs are supported by **reliable and functionally relevant citations**. Traditional Retrieval-Augmented Generation (RAG) systems enhance citation quality by integrating source retrieval, but they treat citations as uniform references, overlooking the **specific role or function** that each citation fulfills within scholarly communication.

This project introduces a **Function-Aware Citation Retrieval and Generation pipeline** designed to address this gap. Instead of relying solely on content similarity, our approach models the *function* of a citation—whether it provides background, compares methods, signals future work, demonstrates usage, or motivates a study. By aligning retrieval and generation processes with citation functions, the system produces references that are not only accurate but also contextually meaningful.

The pipeline consists of four key components:

1. **Text Generation** – An LLM generates coherent, contextually relevant responses to user queries.  
2. **Citation Function Classification** – A specialized model categorizes citations into functional roles such as *background*, *uses*, *compares*, *continuation*, *future*, or *motivation*.  
3. **Function-Aware Retrieval** – Instead of retrieving from a flat index, documents are retrieved from **function-specific indexes**, ensuring more precise alignment between user needs and source material.  
4. **Function-Aware Citation Generation** – Retrieved citations are seamlessly integrated into the LLM’s output in a way that respects their intended role, producing responses with citations that are trustworthy, purposeful, and scholarly.

By combining LLMs with function-aware retrieval strategies, this project aims to **elevate the reliability, interpretability, and trustworthiness of AI-generated scientific text**. Beyond research papers, the methods developed here have broader implications for any domain where citations or references must carry nuanced meaning, such as legal reasoning, policy documents, and technical standards.

This repository contains the source code, models, and experimental setup for building and evaluating function-aware citation systems. We invite researchers and practitioners to explore, extend, and apply this work to advance the next generation of **transparent, citation-grounded AI systems**.

---

## Features
- Function-aware citation classification
- Function-specific retrieval indexes
- Citation-grounded text generation
- Modular pipeline for experimentation

## Getting Started
```bash
# clone the repository
git clone https://github.com/anushkajoshi909/function-aware-citation.git
cd function-aware-citation

# create env & install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
---

## Datasets

### Corpus (unarXive_2024)
- Hugging Face: `ines-besrour/unarxive_2024` (https://huggingface.co/datasets/ines-besrour/unarxive_2024/tree/main)
- Download the corpus; to reproduce our runs you only need these shards:
```
processed_unarxive_extended_data/unarXive_01981221/01/arXiv_src_0101_001.jsonl
processed_unarxive_extended_data/unarXive_01981221/01/arXiv_src_0102_001.jsonl
processed_unarxive_extended_data/unarXive_01981221/01/arXiv_src_0103_001.jsonl
processed_unarxive_extended_data/unarXive_01981221/01/arXiv_src_0104_001.jsonl
```

### Index dataset (FACET / FAISS+E5)
- FAISS index over E5 embeddings (title+abstract+metadata).
- Existing index dirs used in the repo: `e5_index_subset_1/`.

---

## Evaluation Data

### Synthetic (LLM-generated) / FACET( gold)
- Already included at: `RetrievalAugmentedGeneration/synthetic_dataset_strict1.jsonl`
- To regenerate with new questions (200 papers → 200 questions with 400 function + support tags). You can run the following commands:
```bash
cd RetrievalAugmentedGeneration
python3 Data_generation_support.py
```

### Human-annotated subset
- Existing file: `Human_annotation/annotation_batch.csv`
- To create your own sheet for annotation:
```bash
cd Human_annotation
python3 generate_annotation_sheet.py \
  --dataset /data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/RetrievalAugmentedGeneration/synthetic_dataset_strict1.jsonl \
  --out_csv /data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/Human_annotation/annotation_batch.csv \
  --n_papers 50
```

---

## Pipeline
**Classify → Retrieve → Function-filter → Synthesize**

Stable I/O per stage:

- **Function classification** → `{"query","predicted_functions","scores"}`  
- **Semantic retrieval** (E5+FAISS) → `{"query_id","candidates":[{"paper_id","score"}...]}`  
- **Function-aware filtering** → `{"paper_id","supports","fit","topicality","quote","why","core_hits","score"}`  
- **Answer synthesis** → `{"answer","citations":["paper_id",...],"rationales"}`
  
When you run the fillowing command, this will prompt tyou to enter the model name and the question when running the pipeline.
Run end-to-end:
```bash
python3 run_pipeline.py
```

---

## Evaluation (compute metrics vs. gold)
You can rename the eval_runs folder as needed. In this example, the evaluation was performed using the gpt_teuken model. The results for our run are available in the master branch under folders named eval_runs_<model_name>.

This file runs the pipeline for the whole dataset and evaludates against the gold set and provides the scores.

```bash
python3 eval_runner.py \
  --dataset /data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/RetrievalAugmentedGeneration/synthetic_dataset_strict1.jsonl \
  --pipeline /data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/run_pipeline.py \
  --project-root /data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project \
  --runs-dir /data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/eval_runs_3_gpt_teuken \
  --model-index 7 \
  --limit 200
```

## Run Components Individually

You can also run the components of the pipeline individually. Detailed commands are given below.

### 1) Function classification
```bash
cd RetrievalAugmentedGeneration
python3 classifying_question.py
```

### 2) Retrieval (ad-hoc / inspection)
Open the notebook and runt the blocks.
```
Retreival_query_based.ipynb
```

### 3) Function-aware filtering + answer generation
```bash
python3 function_based_answer.py --debug --max-check 10
# --max-check: top-k candidates to score per query
```

---

## Baseline: Standard RAG (abstract-only)

**Scope:** No function awareness. Retrieves top-k abstracts (k=10) and prompts the model to cite **one** paper.

**Metrics:**  
- **Citation Accuracy@1** – cited == gold  
- **Retrieval Recall@10** – gold in retrieved top-k

**Command:**
```bash
python3 baseline_rag_citation_eval.py \
  --dataset /data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/RetrievalAugmentedGeneration/synthetic_dataset_strict1.jsonl \
  --runs-dir /data/horse/ws/anpa439f-Function_Retrieval_Citation/Research_Project/eval_runs_baseline \
  --topk 10 \
  --retrieval-topk 10 \
  --model openai/gpt-oss-120b \
  --limit 200
```

---

## Scripts • Inputs • Output

| Stage / Role                  | Script/Path                                                | Input (key fields)                                         | Output artifact                          | Notes |
|------------------------------|------------------------------------------------------------|-------------------------------------------------------------|------------------------------------------|-------|
| Function classification      | `RetrievalAugmentedGeneration/classifying_question.py`     | `query`                                                     | `classified_output.jsonl`                | `{query, predicted_functions[]}` few-shot over {Background, Uses, Compares, Extends, FutureWork}. |
| Semantic retrieval           | `RetrievalAugmentedGeneration/retrieval_query_based.py` *(nb: `Retreival_query_based.ipynb`)* | `query`; FAISS index in `e5_index_subset_1/`                | `topk_candidates_query.jsonl`            | `{paper_id, title, abstract, cosine}` (E5 over title+abstract+metadata). |
| Function-aware filtering     | `function_based_answer.py`                                 | top-k candidates + `query`                                  | `scored_candidates.jsonl`                | `{paper_id, supports:bool, fit:float, topicality:float, quote, why, core_hits}`. |
| Answer synthesis             | `function_based_answer.py`                                 | supported candidates                                        | `final_answer.jsonl`                     | `{answer, citations[paper_id], rationale}` (evidence-conditioned). |
| Orchestration                | `run_pipeline.py` *(or `run_pipeline_2.py`)*               | `.env`/config, CLI flags                                    | `pipeline_stdout.txt` + artifacts        | Deterministic stage execution, logs prompts, outputs, timestamps. |
| Evaluation                   | `eval_runner.py`                                           | `final_answer.jsonl` + gold                                 | `eval_results.jsonl`                     | Reports Support P/R/F1, Function classification accuracy, `Citation Recall@1`, `Citation Recall@10`. |
| Index build (offline)        | `build_index.py`                                           | `papers.jsonl` (`paper_id`, title, authors, year, abstract) | `e5_index_subset_1/` (FAISS + metadata)  | Concatenate fields; embed with `intfloat/e5-small-v2`. |
| Baseline RAG                 | `baseline_rag_citation_eval.py`                            | synthetic dataset + retriever                               | `baseline_eval.json` (in run dir)        | Citation Recall@1 and Citation Recall@10 (no function signals). |

---

## Repo Pointers
- `Human_annotation/` – scripts & CSVs for human labels  
- `RetrievalAugmentedGeneration/` – classification, retrieval, generation, synthetic data  
- `eval_runs_*` – experiment outputs (deepseek, gpt_teuken, llama4, baseline, etc.)  
- `e5_index_subset_1/` – FAISS/E5 indices  
- Notebooks: `Load_dataset.ipynb`, `Retreival_query_based.ipynb`, `function_aware_e5_index.ipynb`  
- Runners: `run_pipeline.py`, `eval_runner.py`, `baseline_rag_citation_eval.py`

## Citation
If you use this code or dataset setup, please cite:
```
Joshi, A. (2025). Function-Aware Citation Retrieval and Generation. GitHub: anushkajoshi909/function-aware-citation
```

