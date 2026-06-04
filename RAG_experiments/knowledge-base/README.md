# Personal Knowledge Base — RAG with Claude

A local RAG (Retrieval-Augmented Generation) application. Drop in PDFs, ask Claude questions, and get answers grounded in your documents.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your Anthropic API key

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
# then edit .env and replace "your-key-here" with your actual key
```

### 3. Drop PDFs into docs/

```
docs/
  my-paper.pdf
  company-handbook.pdf
  ...
```

---

## Usage

### Step 1 — Ingest documents

Loads PDFs, chunks them, embeds with a local model, and saves to ChromaDB.

```bash
python ingest.py
```

Re-run this whenever you add new documents.

### Step 2a — CLI chat

```bash
python query.py
```

Type questions, get answers with source citations. Type `quit` to exit.

### Step 2b — Streamlit UI

```bash
streamlit run app.py
```

Opens a browser chat interface at `http://localhost:8501`.

---

## How RAG works in this project

RAG = **Retrieve** relevant context, then **Augment** the LLM prompt with it, then **Generate** an answer.

```
Your PDFs
   ↓ (ingest.py)
Text chunks  →  Embeddings (all-MiniLM-L6-v2)  →  ChromaDB (vector store)

User question
   ↓ (query.py / app.py)
Question embedding  →  Similarity search in ChromaDB  →  Top 4 chunks
   ↓
Chunks injected into prompt  →  Claude (claude-sonnet-4-20250514)  →  Answer + sources
```

See the `../docs/` folder for detailed explanations of each step.

---

## Project structure

```
knowledge-base/
├── docs/           ← drop your PDFs here
├── chroma_db/      ← auto-created; holds the vector index (gitignored)
├── ingest.py       ← ingestion pipeline
├── query.py        ← CLI chat
├── app.py          ← Streamlit UI
├── requirements.txt
└── .env.example
```
