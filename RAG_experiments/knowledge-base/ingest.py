"""
ingest.py - Load PDFs from docs/, chunk them, embed, and store in ChromaDB.

Run this once (and re-run whenever you add new documents):
    python ingest.py
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def ingest():
    # 1. Load all PDFs from docs/
    print(f"Loading PDFs from '{DOCS_DIR}/'...")
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    documents = loader.load()

    if not documents:
        print("No PDF documents found. Drop some PDFs into the docs/ folder and re-run.")
        return

    print(f"Loaded {len(documents)} page(s) from {len(set(d.metadata['source'] for d in documents))} file(s).")

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    # 3. Embed using a local HuggingFace model (no API key needed)
    print(f"Loading embedding model '{EMBED_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # 4. Store embeddings in ChromaDB (persisted to chroma_db/)
    print(f"Embedding chunks and persisting to '{CHROMA_DIR}/'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print(f"\nDone! {len(chunks)} chunks stored in '{CHROMA_DIR}/'.")
    print("You can now run query.py or app.py to chat with your documents.")


if __name__ == "__main__":
    ingest()
