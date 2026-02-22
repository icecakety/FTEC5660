# Reproducibility Study: RAG-based Document Q&A Agent with Gemini (Vertex AI)

**Course:** Agentic AI for Business and FinTech (FTEC5660)  
**Author:** [HU TAINYU 1155249527]  
**Date:** 2026-02-22  

---

## Project Summary

This repository contains a reproducibility study of an agentic RAG (Retrieval-Augmented Generation) system for financial document question answering. The original project (instructor-provided) uses a LangChain-based pipeline with Gemini via Vertex AI. We reproduce a **factual QA accuracy** result on a small financial dataset (8 questions) and then apply a controlled modification: **increasing the LLM temperature from 0.1 (baseline) to 0.7**. The goal is to verify the reproducibility of the reported result and measure the impact of temperature on answer accuracy.

The implementation avoids deprecated `langchain.chains` and uses a **manual RAG function** with:
- `HuggingFaceEmbeddings` (all-MiniLM-L6-v2)
- FAISS vector store
- Gemini 2.5 Flash (via Vertex AI) as the LLM

All code is provided in the Jupyter notebook [`notebook_reporduce.ipynb`]((https://colab.research.google.com/drive/1IYUfQf6CEgYAgIGjSti-o3XZYRwpVcCV?usp=sharing)).

---

## Reproduction Target

- **Claim reproduced:** The agent achieves **100% accuracy** on a set of 8 factual/financial questions about five companies (Apple, Microsoft, Amazon, Tesla, NVIDIA).  
- **Metric:** Accuracy = (number of correct answers) / (total questions). A prediction is considered correct if the expected answer string appears (case-insensitive) in the generated response.  
- **Dataset:** Hand-crafted financial profiles (see notebook cell 6).

---

## Setup Notes

### Environment
- **Platform:** Google Colab (Python 3.10, optional T4 GPU)
- **Dependencies:** See [`notebook_reporduce.ipynb`](https://colab.research.google.com/drive/1IYUfQf6CEgYAgIGjSti-o3XZYRwpVcCV?usp=sharing) (install cell). Key packages:
  - `langchain-google-vertexai`
  - `langchain-community`
  - `faiss-cpu`
  - `sentence-transformers`
  - `google-cloud-aiplatform`

### Authentication (Vertex AI)
1. You need a Google Cloud project with **Vertex AI API enabled**.
2. In Colab, run the authentication cell – it will prompt you to authorize access.
3. Set your `PROJECT_ID` and `LOCATION` (default: `us-central1`) in the notebook.

### Compute
- The embedding model runs on CPU; the Gemini API calls incur no local GPU cost.
- Total runtime: ~5 minutes (including model downloads and API calls).

---

## How to Run

1. **Open the notebook** in Google Colab (or Jupyter with the same dependencies).
2. **Execute all cells sequentially**. The notebook is self-contained:
   - Installs required packages.
   - Authenticates and initializes Vertex AI.
   - Creates a sample financial dataset (`financial_data.txt`).
   - Builds a FAISS vector store with embeddings.
   - Defines a manual RAG function.
   - Runs baseline evaluation (temperature=0.1).
   - Compares reproduced result with the target (100%).
   - Logs a debug diary.
   - Creates a modified LLM (temperature=0.7) and re-evaluates.
   - Compares baseline vs. modified results.
   - Exports summary files (CSV, JSON, Markdown).

3. **Check outputs**: The notebook saves:
   - `baseline_results.csv` – per‑question results (t=0.1)
   - `modified_results.csv` – per‑question results (t=0.7)
   - `reproducibility_summary.json` – high‑level metrics
   - `report_summary.md` – a brief Markdown report

> **Note:** The notebook contains a real `PROJECT_ID`; replace it with your own before running.

---

## Results

### Baseline (Temperature = 0.1)

| Metric               | Value     |
|----------------------|-----------|
| Accuracy             | 100.0%    |
| Correct Answers      | 8 / 8     |
| Avg Response Time    | 0.62 s    |

The reproduced result exactly matches the target claim of 100% accuracy.

### Modification (Temperature = 0.7)

| Metric               | Value     |
|----------------------|-----------|
| Accuracy             | 100.0%    |
| Correct Answers      | 8 / 8     |
| Avg Response Time    | 0.63 s    |

### Comparison

| Metric               | Baseline | Modified | Change |
|----------------------|----------|----------|--------|
| Temperature          | 0.1      | 0.7      | +0.6   |
| Accuracy (%)         | 100.0    | 100.0    | 0.0    |
| Correct Answers      | 8        | 8        | 0      |
| Avg Response Time (s)| 0.62     | 0.63     | +0.01  |

**Detailed Question‑Level Outcome:**  
All questions were answered correctly in both runs, although the higher‑temperature run sometimes produced slightly more verbose or varied phrasing (e.g., for the Tesla founders question). No factual errors were introduced.

---

## Debug Diary

### Issues Encountered

| Issue | Symptom | Root Cause | Resolution | Status |
|-------|---------|------------|------------|--------|
| **ModuleNotFoundError for `langchain.chains`** | Import error when trying to use `RetrievalQA` | LangChain core modules not fully installed; `langchain-community` does not include all chains. | Switched to a **manual RAG implementation** (retrieval + prompt + LLM call). | ✅ Resolved |
| **Vertex AI authentication** | API calls fail with authentication errors. | Missing user authentication and project initialization. | Added `auth.authenticate_user()` and `vertexai.init()` in Colab. | ✅ Resolved |
| **Embeddings model download warnings** | Deprecation warnings from `HuggingFaceEmbeddings`. | The class is deprecated in `langchain-community`. | Ignored warnings; model still works. Future migration may be needed. | ⚠️ Acceptable |

### Performance Observations
- **Response time:** ~0.6–0.9 s per query (depends on Vertex AI latency).
- **RAM usage:** ~2 GB (mostly from FAISS index and embeddings model).
- **Disk usage:** ~500 MB (cached models, vector store).

---

## Conclusions

- **Reproducibility:** The claimed 100% accuracy on this small factual dataset is **fully reproducible** under the same setup (temperature=0.1, Gemini 2.5 Flash, the provided retrieval pipeline).
- **Modification impact:** Increasing temperature to 0.7 **did not change accuracy** for this specific set of questions. However, the responses exhibited slightly more variability (e.g., extra clarifications), which could potentially harm precision on more ambiguous or numerical tasks.
- **Key findings:**
  1. Lower temperature (0.1–0.3) is preferable for factual Q&A where exact answers are required.
  2. Manual RAG implementation avoids dependency on unstable LangChain chains and is easy to debug.
  3. The retrieval component (FAISS + all‑MiniLM‑L6‑v2) reliably returns relevant chunks (top‑3) for all questions.
- **Recommendations for future users:**
  - For production factual systems, keep temperature ≤ 0.3.
  - Implement answer validation to catch hallucinations, especially if temperature is increased.
  - Cache embeddings to reduce startup time.
  - If you encounter import errors, consider a manual RAG approach as shown in this notebook.

---

## Repository Contents

- [`notebook_reporduce.ipynb`](./notebook_reporduce.ipynb) – Main Jupyter notebook with all code, results, and analysis.
- `baseline_results.csv` – Per‑question results for temperature=0.1 (generated after run).
- `modified_results.csv` – Per‑question results for temperature=0.7 (generated after run).
- `reproducibility_summary.json` – JSON summary of the study.
- `report_summary.md` – Brief Markdown report (generated automatically).
- `financial_data.txt` – Sample financial dataset (created at runtime).
- `financial_vectorstore/` – FAISS index (created at runtime; not committed to git).

> **Note:** The generated output files are included as examples. Re-running the notebook will overwrite them.

---

## License

This project is for educational purposes as part of FTEC5660. The code is provided under the MIT License (see [LICENSE]((https://opensource.org/license/mit)) file if present).

---

## Acknowledgments

- Original project inspiration: instructor‑provided list (FTEC5660).
- Built with [LangChain](https://www.langchain.com/), [Google Vertex AI](https://cloud.google.com/vertex-ai), and [HuggingFace](https://huggingface.co/) embeddings.
