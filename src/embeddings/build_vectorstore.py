import gc
import shutil
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.ingestion.chunk_text import chunk_documents
from src.ingestion.load_pdfs import load_user_pdfs

_GLOBAL_EMBEDDING_MODEL = None

def get_embedding_model() -> HuggingFaceEmbeddings:
    global _GLOBAL_EMBEDDING_MODEL
    try:
        import streamlit as st
        if hasattr(st, "cache_resource"):
            @st.cache_resource
            def _get_st_embedding_model():
                return HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            return _get_st_embedding_model()
    except Exception:
        pass

    if _GLOBAL_EMBEDDING_MODEL is None:
        _GLOBAL_EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _GLOBAL_EMBEDDING_MODEL

def build_user_vectorstore(user_id: str) -> Chroma:
    docs = load_user_pdfs(user_id)
    chunks = chunk_documents(docs)

    embedding_model = get_embedding_model()

    persist_dir = Path(f"db/chroma/{user_id}")
    if persist_dir.exists():
        gc.collect()
        shutil.rmtree(persist_dir, ignore_errors=True)

    persist_dir.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        collection_name=f"{user_id}_pdf_collection",
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(persist_dir)
    )

    return vectorstore

def delete_pdf_from_user_vectorstore(user_id: str, file_name: str) -> bool:
    try:
        from src.retrieval.retriever import load_user_vectorstore
        vectorstore = load_user_vectorstore(user_id)
        vectorstore.delete(where={"file_name": file_name})
        del vectorstore
        gc.collect()
        return True
    except Exception as e:
        import logging
        logging.error(f"Failed to delete {file_name} from Chroma vectorstore: {e}")
        return False