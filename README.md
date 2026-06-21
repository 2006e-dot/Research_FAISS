# Financial Sentiment Analysis using Hybrid Retrieval (FAISS + BM25)

## Overview

This repository explores retrieval-based financial sentiment classification using dense and sparse information retrieval techniques.

The objective is to classify financial news headlines into:

* Positive
* Neutral
* Negative

Instead of using traditional machine learning classifiers, this work investigates whether sentiment can be inferred from retrieved historical financial headlines using semantic and lexical retrieval methods.

---

## Dataset

A custom financial sentiment corpus was constructed by merging multiple financial news datasets:

* SEntFiN Dataset
* NSE Financial News Corpus
* Additional curated financial headlines

Duplicate headlines were removed before experimentation.

### Final Dataset Statistics

| Label    | Count |
| -------- | ----: |
| Positive |  6205 |
| Neutral  |  5621 |
| Negative |  5067 |

**Total Headlines:** 16,893

---

## Methodology

### 1. FAISS Retrieval

FAISS (Facebook AI Similarity Search) is used as the dense retrieval component.

* Embedding Model: all-MiniLM-L6-v2
* Similarity Metric: Inner Product (Cosine Similarity after normalization)
* Retrieval Type: Semantic Search

---

### 2. BM25 Retrieval

BM25 is used as the sparse retrieval component.

* Token-based ranking
* Lexical matching
* Traditional Information Retrieval baseline

---

### 3. Hybrid Retrieval (FAISS + BM25)

The hybrid system combines:

* Top-K FAISS Results
* Top-K BM25 Results

Retrieved documents are merged and sentiment prediction is obtained through majority voting over retrieved labels.

---

### 4. Reciprocal Rank Fusion (RRF)

RRF combines rankings from:

* FAISS
* BM25

using:

Score(d) = Σ 1 / (k + rank(d))

The fused ranking is then used for sentiment prediction.

---

## Experimental Results

### Baseline Comparison

| Method                | Accuracy |
| --------------------- | -------: |
| FAISS                 |   69.87% |
| BM25                  |   72.57% |
| Hybrid (FAISS + BM25) |   74.13% |
| FAISS + BM25 + RRF    |   73.39% |

---

### Detailed Performance Comparison

| Method | Precision | Recall | F1 Score | Accuracy |
| ------ | --------: | -----: | -------: | -------: |
| FAISS  |      0.71 |   0.70 |     0.70 |   69.87% |
| BM25   |      0.73 |   0.73 |     0.73 |   72.57% |
| Hybrid |      0.74 |   0.74 |     0.74 |   74.13% |
| RRF    |      0.73 |   0.73 |     0.73 |   73.39% |

---

## 5-Fold Cross Validation (Hybrid Retrieval)

| Fold   | Accuracy |
| ------ | -------: |
| Fold 1 |   74.37% |
| Fold 2 |   74.70% |
| Fold 3 |   74.79% |
| Fold 4 |   74.22% |
| Fold 5 |   75.22% |

### Final Cross Validation Results

| Metric             |  Value |
| ------------------ | -----: |
| Mean Accuracy      | 74.66% |
| Standard Deviation |  0.35% |

The low standard deviation indicates stable and consistent performance across different train-test splits.

---

## Key Findings

* BM25 outperformed dense retrieval (FAISS) on the financial headline dataset.
* Combining sparse and dense retrieval improved classification accuracy.
* Hybrid retrieval achieved the best overall performance.
* RRF improved ranking diversity but did not outperform the hybrid voting strategy.
* Cross-validation confirmed the robustness and stability of the hybrid retrieval approach.

---

## Repository Structure

```text
.
├── MASTER_FINANCIAL_CORPUS_V1.csv
├── faiss_only.py
├── faiss+bm25.py
├── faiss+bm25+rrf.py
├── RESULTS.txt
├── cross_validation.py
└── README.md
```

---

## Future Work

Planned extensions include:

* Maximum Marginal Relevance (MMR)
* Retrieval-Augmented Generation (RAG)
* RAG Fusion
* LLaMA-based sentiment reasoning
* Comparative analysis of retrieval fusion techniques
* Financial domain-specific embedding models

---


Research Interests:

* Financial Sentiment Analysis
* Information Retrieval
* Retrieval-Augmented Generation (RAG)
* Natural Language Processing
* Machine Learning

---

