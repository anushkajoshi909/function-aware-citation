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
