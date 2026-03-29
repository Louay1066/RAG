import os
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_document(file_path: str):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    if ext in (".txt", ".md"):
        return TextLoader(file_path, encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")


def ingest_documents(
    file_paths: list[str],
    chroma_dir: str,
    api_key: str,
    embedding_model: str = "text-embedding-3-small",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> tuple[int, int]:
    """Load, chunk, embed, and store documents. Returns (num_pages, num_chunks)."""
    all_docs = []
    for path in file_paths:
        all_docs.extend(load_document(path))

    if not all_docs:
        return 0, 0

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ).split_documents(all_docs)

    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=api_key)

    if os.path.exists(chroma_dir):
        vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
        vectorstore.add_documents(chunks)
    else:
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=chroma_dir,
        )

    return len(all_docs), len(chunks)


def clear_vectorstore(chroma_dir: str):
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)


def get_collection_count(chroma_dir: str) -> int:
    """Return the number of chunks currently stored in ChromaDB."""
    if not os.path.exists(chroma_dir):
        return 0
    try:
        import chromadb
        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.get_collection("langchain")
        return collection.count()
    except Exception:
        return 0
