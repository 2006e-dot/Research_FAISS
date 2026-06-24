# Financial Sentiment Classification using Hybrid Retrieval, BGE-M3 and MMR

## Overview

This project explores retrieval-based financial sentiment classification using sparse and dense retrieval techniques instead of traditional supervised machine learning classifiers.

The objective is to classify financial news headlines into:

* Positive
* Neutral
* Negative

Rather than training a classifier, sentiment is inferred from previously seen financial headlines retrieved using Information Retrieval techniques.

The project progressively evaluates:

* BM25
* FAISS
* Hybrid Retrieval (BM25 + FAISS)
* Reciprocal Rank Fusion (RRF)
* BAAI/bge-m3 embeddings
* Maximum Marginal Relevance (MMR)

and studies how retrieval quality impacts sentiment classification performance.

---

## Dataset

A custom financial sentiment corpus was constructed by combining multiple publicly available financial news datasets.

Sources include:

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

## Problem Statement

Given a financial headline:

```text
Apple shares surge after strong quarterly earnings
```

retrieve the most relevant historical financial headlines and infer sentiment using the labels of retrieved documents.

This converts sentiment classification into a retrieval problem.

---

# Methodology

## 1. Dense Retrieval (FAISS)

FAISS is used as the semantic retrieval engine.

### Embedding Model

Initially:

```text
all-MiniLM-L6-v2
```

Later experiments:

```text
BAAI/bge-m3
```

### Similarity Metric

```text
Cosine Similarity
```

implemented using:

```text
FAISS IndexFlatIP
```

with normalized embeddings.

---

## 2. Sparse Retrieval (BM25)

BM25 serves as the lexical retrieval component.

Characteristics:

* Exact keyword matching
* Traditional Information Retrieval baseline
* No embedding generation required

---

## 3. Hybrid Retrieval

The hybrid system combines:

```text
BM25 Top-K
+
FAISS Top-K
```

Retrieved documents are merged and sentiment is predicted using majority voting over retrieved labels.

Pipeline:

```text
Query
↓
BM25 Retrieval
+
FAISS Retrieval
↓
Merge Results
↓
Majority Voting
↓
Sentiment Prediction
```

---

## 4. Reciprocal Rank Fusion (RRF)

RRF combines rankings from BM25 and FAISS.

Formula:

```text
RRF(d) = Σ 1 / (k + rank(d))
```

where:

* d = document
* rank(d) = document rank
* k = ranking constant

The highest scoring documents are selected for classification.

---

## 5. Maximum Marginal Relevance (MMR)

MMR improves retrieval diversity by reducing redundancy among retrieved documents.

Formula:

```text
MMR =
λ × Relevance
-
(1 - λ) × Diversity
```

where:

* Relevance measures similarity to the query
* Diversity penalizes near-duplicate retrieved documents

Pipeline:

```text
BM25 Top 20
+
FAISS Top 20
↓
Union
↓
MMR
↓
Top 5 Diversified Documents
↓
Majority Voting
```

Configuration:

```text
λ = 0.7
```

---

# Experimental Results

## Baseline Retrieval Performance

| Method                 | Accuracy |
| ---------------------- | -------: |
| FAISS (MiniLM)         |   69.87% |
| BM25                   |   72.57% |
| Hybrid (MiniLM + BM25) |   74.13% |
| Hybrid + RRF           |   73.39% |

---

## 5-Fold Cross Validation

### Hybrid Retrieval (MiniLM + BM25)

| Fold   | Accuracy |
| ------ | -------: |
| Fold 1 |   74.37% |
| Fold 2 |   74.70% |
| Fold 3 |   74.79% |
| Fold 4 |   74.22% |
| Fold 5 |   75.22% |

### Final Result

| Metric             |  Value |
| ------------------ | -----: |
| Mean Accuracy      | 74.66% |
| Standard Deviation |  0.35% |

---

## Hybrid + RRF Cross Validation

| Fold   | Accuracy |
| ------ | -------: |
| Fold 1 |   72.21% |
| Fold 2 |   72.36% |
| Fold 3 |   72.48% |
| Fold 4 |   72.76% |
| Fold 5 |   73.56% |

### Final Result

| Metric             |  Value |
| ------------------ | -----: |
| Mean Accuracy      | 72.68% |
| Standard Deviation |  0.48% |

---

## BGE-M3 Hybrid Retrieval

The MiniLM encoder was replaced with:

```text
BAAI/bge-m3
```

while keeping the hybrid retrieval framework unchanged.

### 5-Fold Cross Validation

| Fold   | Accuracy |
| ------ | -------: |
| Fold 1 |   79.11% |
| Fold 2 |   80.20% |
| Fold 3 |   79.14% |
| Fold 4 |   78.83% |
| Fold 5 |   79.96% |

### Final Result

| Metric             |  Value |
| ------------------ | -----: |
| Mean Accuracy      | 79.45% |
| Standard Deviation |  0.53% |

---

# Best Performing Model

## Hybrid Retrieval + MMR

Embedding Model:

```text
BAAI/bge-m3
```

Retrieval Strategy:

```text
BM25 Top 20
+
FAISS Top 20
↓
Union
↓
MMR (λ = 0.7)
↓
Top 5 Documents
↓
Majority Voting
```

### 5-Fold Cross Validation

| Fold   | Accuracy |
| ------ | -------: |
| Fold 1 |   81.89% |
| Fold 2 |   83.10% |
| Fold 3 |   82.33% |
| Fold 4 |   81.32% |
| Fold 5 |   82.42% |

### Final Result

| Metric             |  Value |
| ------------------ | -----: |
| Mean Accuracy      | 82.21% |
| Standard Deviation |  0.59% |

---

## Performance Progression

| Method                       |   Accuracy |
| ---------------------------- | ---------: |
| FAISS (MiniLM)               |     69.87% |
| BM25                         |     72.57% |
| Hybrid (MiniLM + BM25)       |     74.66% |
| Hybrid + RRF                 |     72.68% |
| Hybrid (BGE-M3 + BM25)       |     79.45% |
| Hybrid + MMR (BGE-M3 + BM25) | **82.21%** |

---

# Key Findings

* BM25 outperformed dense retrieval using MiniLM embeddings.
* Hybrid retrieval consistently outperformed standalone retrieval methods.
* Reciprocal Rank Fusion did not improve classification performance on this dataset.
* Replacing MiniLM with BGE-M3 produced a significant improvement in retrieval quality.
* Maximum Marginal Relevance reduced retrieval redundancy and improved classification accuracy.
* The best configuration achieved 82.21% accuracy without training a dedicated classifier.
* Cross-validation results demonstrate stable performance across different train-test splits.

---

# Repository Structure

```text
.
├── MASTER_FINANCIAL_CORPUS_V1.csv
├── README.md
├── RESULTS.txt
│
├── faiss_only.py
├── faiss+bm25.py
├── faiss+bm25+rrf.py
│
├── cross_validation.py
├── cross_validation_bgem3.py
├── cross_validation_bgem3+mmr.py
│
├── faiss+bm25+rrf_cross_validation.py
```

---

# Future Work

Planned future experiments include:

* Weighted Voting using Retrieval Scores
* RRF + MMR Hybrid Fusion
* Vectorless RAG Fusion
* Retrieval-Augmented Generation (RAG)
* LangChain-based Hybrid Retrieval Pipelines
* Graph RAG for Financial Knowledge Graphs
* Financial Domain-Specific Embedding Models
* LLaMA-based Sentiment Reasoning
* Comparative Analysis of Retrieval Fusion Techniques

---

# Research Areas

* Financial Sentiment Analysis
* Information Retrieval
* Retrieval-Augmented Generation (RAG)
* Hybrid Search Systems
* Natural Language Processing
* Machine Learning
* Large Language Models

---

## Author

Ayush Bagdai

Student, IIIT Vadodara

Interests:

* Retrieval-Augmented Generation
* Information Retrieval
* Financial NLP
* Large Language Models
* Machine Learning Research
