from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.ingestion.chunk_text import chunk_documents
from src.ingestion.load_pdfs import load_user_pdfs

def build_user_vectorstore(user_id: str) -> Chroma:
    docs = load_user_pdfs(user_id)
    chunks = chunk_documents(docs)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    persist_dir = Path(f"db/chroma/{user_id}")
    if persist_dir.exists():
        import shutil
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
        return True
    except Exception as e:
        import logging
        logging.error(f"Failed to delete {file_name} from Chroma vectorstore: {e}")
        return False