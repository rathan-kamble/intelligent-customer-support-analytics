# Intelligent Customer Support Analytics and Knowledge Assistant

> End-to-end AI system for customer support operations using SparkML for ticket priority prediction and a RAG pipeline for intelligent knowledge assistance.

---

## Project Overview

This project modernizes customer support operations by combining scalable machine learning with large language model capabilities. It was built as part of Module 9 — ML Pipeline Creation at Scale using SparkML and LLM Application Engineering.

The system solves two core business problems:

1. **Automatic ticket priority prediction** — classifies incoming support tickets as Low, Medium, High, or Critical so urgent cases are handled faster.
2. **RAG-powered knowledge assistant** — answers agent queries using internal knowledge base documents such as FAQs, troubleshooting guides, refund policies, and escalation SOPs.

---

## Architecture

```
Part A — SparkML Pipeline
─────────────────────────────────────────────────────
Synthetic Dataset (5,000 tickets)
        │
        ▼
Spark Ingestion + Schema Definition
        │
        ▼
Feature Engineering
  ├── StringIndexer (categorical columns)
  ├── OneHotEncoder
  ├── Tokenizer → StopWordsRemover → HashingTF → IDF (ticket text)
  ├── VectorAssembler
  └── StandardScaler
        │
        ▼
Model Training
  ├── Random Forest (Multiclass — 4 classes)
  └── Gradient Boosted Trees (Binary — Urgent vs Not Urgent)
        │
        ▼
Evaluation
  ├── Accuracy, F1, Precision, Recall
  ├── Confusion Matrix
  └── Feature Importance

Part B — RAG Pipeline
─────────────────────────────────────────────────────
Knowledge Base Documents (5 documents)
        │
        ▼
Document Chunking (chunk_size=500, overlap=50)
        │
        ▼
Embeddings (sentence-transformers/all-MiniLM-L6-v2)
        │
        ▼
Vector Store (ChromaDB)
        │
        ▼
Retriever (Top-3 similarity search)
        │
        ▼
LLM (GPT-3.5-turbo) → Grounded Answer

MLOps Layer
─────────────────────────────────────────────────────
MLflow Experiment Tracking
  ├── Random Forest run
  ├── GBT run
  └── RAG Pipeline run
Model Registry (Staging / Production)
RAG Monitoring Logs
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Distributed ML | Apache Spark 4.0 + SparkML |
| Experiment Tracking | MLflow |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| LLM | OpenAI GPT-3.5-turbo |
| RAG Framework | LangChain |
| Data Processing | PySpark, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Environment | Google Colab |

---

## Results

### Part A — SparkML Models

| Model | Task | Accuracy | F1 Score | AUC-ROC |
|---|---|---|---|---|
| Random Forest | Multiclass (4 classes) | 69.7% | 0.694 | — |
| Gradient Boosted Trees | Binary (Urgent vs Not) | 99.7% | 0.997 | 0.9998 |

**Winner: GBT** — 99.7% accuracy with AUC-ROC of 0.9998

### Part B — RAG Pipeline

| Metric | Value |
|---|---|
| Documents Indexed | 5 |
| Chunks Created | 20 |
| Average Relevance Score | 0.70+ |
| Questions Evaluated | 5 |

---

## Dataset

A synthetic dataset of 5,000 customer support tickets was generated with the following columns:

| Column | Description |
|---|---|
| ticket_id | Unique ticket identifier |
| created_at | Timestamp of ticket creation |
| channel | email / chat / web / call |
| product | ProductA / B / C / D |
| region | North / South / East / West / Central |
| customer_tier | Bronze / Silver / Gold / Platinum |
| issue_category | billing / technical / shipping / returns / account / general |
| ticket_text | Free-text ticket description |
| response_time_hours | Time taken to respond |
| num_previous_tickets | Customer's ticket history count |
| sentiment_score | Sentiment score [-1.0 to 1.0] |
| contains_urgent_words | Binary flag for urgent language |
| priority_label | Low / Medium / High / Critical |

---

## Knowledge Base Documents

| Document | Description |
|---|---|
| faq.txt | Frequently asked questions and answers |
| troubleshooting.txt | Product-specific issue resolutions |
| refund_policy.txt | Return and refund procedures |
| escalation_sop.txt | Escalation standards and procedures |
| warranty_policy.txt | Warranty coverage and claim process |

---

## Project Structure

```
intelligent-customer-support-analytics/
│
├── Module9_Assignment.ipynb     # Main notebook — all code
├── generate_tickets.py          # Synthetic dataset generator
├── tickets.csv                  # Generated dataset (5,000 rows)
├── confusion_matrix_rf.png      # Random Forest confusion matrix
├── feature_importance.png       # Feature importance chart
├── model_registry.json          # MLflow model registry
├── rag_monitoring.json          # RAG query monitoring logs
└── README.md                    # Project documentation
```

---

## How to Run

### 1. Open in Google Colab
Upload `Module9_Assignment.ipynb` to [Google Colab](https://colab.research.google.com)

### 2. Generate the Dataset
```python
python generate_tickets.py
```

### 3. Set Your OpenAI API Key
```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

### 4. Run All Cells
Run the notebook cells in order from top to bottom.

---

## Key Insights

- **Ticket text is the strongest signal** — TF-IDF features from ticket descriptions drive priority prediction more than any other feature.
- **GBT outperforms Random Forest significantly** — 99.7% vs 69.7% accuracy for binary urgency classification.
- **RAG grounds answers reliably** — the pipeline retrieves correct context and generates accurate answers without hallucination.
- **Platinum customers drive escalations** — tier and sentiment score are key factors in priority assignment.

---

## Author

**Raja Rathan Kamble**
Module 9 Assignment — ML Pipeline Creation at Scale

---

## License

This project is licensed under the MIT License.
