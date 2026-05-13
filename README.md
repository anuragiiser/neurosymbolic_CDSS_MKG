# Neurosymbolic Integration of Medical Knowledge Graphs for Clinical Decision Support

This repository contains the implementation and experimental code for the paper:

**“Neurosymbolic Integration of Medical Knowledge Graphs for Performance Enhancement in Clinical Decision Support”**

The project explores how structured medical knowledge graphs can enhance the reasoning capability of medical Large Language Models (LLMs) for diagnosis prediction and clinical decision support tasks.

---

## Overview

Large Language Models (LLMs) have shown strong performance in clinical reasoning tasks using Electronic Health Records (EHRs) and clinical notes. However, they often lack explicit structured medical knowledge.

This repository implements a **neurosymbolic framework** that augments medical LLMs with information retrieved from Medical Knowledge Graphs (MKGs), improving diagnostic reasoning across multiple prompting strategies such as:

- Zero-shot prompting
- Few-shot prompting
- RAG

In addition to using existing MKGs, this work also introduces a **PubMed-based Medical Knowledge Graph (PMKG)** integrating:

- PubMed knowledge
- ICD-9 codes
- UMLS concepts
- Symptom–disease relationships

---

## Repository Structure

```text
.
├── PMKG/
│   └── Code for constructing the PubMed-based Medical Knowledge Graph
│
├── llm_inferences/
│   └── Code for running LLM-based inference experiments
│
├── dataset/
│
└── README.md
```

---

## Experimental Evaluation

The framework was evaluated on:

- MIMIC-III
- MIMIC-IV

using multiple medical LLMs and prompting strategies.

Results demonstrate consistent improvements in diagnostic performance through knowledge graph augmentation.

---
