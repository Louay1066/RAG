<div align="center">

# RAG Pipeline

**Upload. Ask. Evaluate.**

A complete Retrieval-Augmented Generation pipeline with a clean web interface — ingest your documents, query them with natural language, and evaluate retrieval quality with LLM-as-judge scoring.

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1c3c3c?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)

---

</div>

## Features

| Feature | Description |
|---|---|
| **Document Ingestion** | Upload PDF, TXT, and Markdown files — automatically chunked, embedded, and stored |
| **Semantic Search & QA** | Ask questions in plain English and get grounded answers with source citations |
| **Pipeline Evaluation** | Score your pipeline on faithfulness, relevancy, context quality, and correctness |
| **Configurable** | Tune chunk size, overlap, top-K, embedding model, and LLM from the sidebar |
| **Persistent Store** | ChromaDB vector store persists across sessions |

---

## Architecture

```
                  ┌──────────────┐
                  │   Streamlit  │
                  │   Web  UI    │
                  └──────┬───────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    ┌──────────┐  ┌──────────┐  ┌──────────────┐
    │  Ingest  │  │  Query   │  │  Evaluate    │
    │          │  │          │  │              │
    │ Load     │  │ Retrieve │  │ Faithfulness │
    │ Chunk    │  │ Generate │  │ Relevancy    │
    │ Embed    │  │ Answer   │  │ Context      │
    │ Store    │  │          │  │ Correctness  │
    └────┬─────┘  └────┬─────┘  └──────┬───────┘
         │             │               │
         ▼             ▼               ▼
    ┌─────────┐  ┌──────────┐   ┌──────────┐
    │ChromaDB │  │ OpenAI   │   │ OpenAI   │
    │         │◄─┤Embeddings│   │ LLM      │
    │ Vectors │  │ + LLM    │   │ (Judge)  │
    └─────────┘  └──────────┘   └──────────┘
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/RAG.git
cd RAG
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key

```bash
cp .env.example .env
```

Open `.env` and paste your OpenAI API key:

```
OPENAI_API_KEY=sk-your-key-here
```

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## Usage

### Documents Tab

1. Drag and drop your files (PDF, TXT, or MD)
2. Click **Ingest into Vector Store**
3. The status panel shows how many chunks are stored

### Query Tab

1. Type your question
2. Click **Ask**
3. View the answer and the retrieved context chunks with relevance scores

### Evaluate Tab

1. Enter test questions manually **or** paste / upload a JSON test set:

```json
[
  { "question": "What is RAG?", "ground_truth": "RAG stands for..." },
  { "question": "How does chunking work?" }
]
```

2. Click **Run Evaluation**
3. Review scores across four metrics:

| Metric | What it measures |
|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **Answer Relevancy** | Does the answer address the question? |
| **Context Relevancy** | Are the retrieved chunks relevant to the question? |
| **Correctness** | Does the answer match the ground truth? *(requires ground truth)* |

---

## Configuration

All settings are available in the sidebar:

| Setting | Default | Description |
|---|---|---|
| Embedding model | `text-embedding-3-small` | OpenAI embedding model |
| LLM model | `gpt-4o-mini` | Model used for answer generation and evaluation |
| Chunk size | 1000 | Characters per chunk |
| Chunk overlap | 200 | Overlap between consecutive chunks |
| Top K | 4 | Number of chunks retrieved per query |

---

## Project Structure

```
RAG/
├── app.py                 # Streamlit web UI
├── rag/
│   ├── __init__.py
│   ├── ingest.py          # Document loading, chunking, embedding
│   ├── query.py           # Retrieval + LLM generation
│   └── evaluate.py        # LLM-as-judge evaluation metrics
├── requirements.txt
├── .env.example           # API key template
└── .gitignore
```

---

## Tech Stack

- **[LangChain](https://langchain.com)** — Document loading, text splitting, chains
- **[ChromaDB](https://www.trychroma.com)** — Local vector storage
- **[OpenAI](https://openai.com)** — Embeddings (`text-embedding-3-small`) and LLM (`gpt-4o-mini`)
- **[Streamlit](https://streamlit.io)** — Web interface
- **[Pandas](https://pandas.pydata.org)** — Evaluation results display

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
<sub>Built with LangChain, ChromaDB, and Streamlit</sub>
</div>
