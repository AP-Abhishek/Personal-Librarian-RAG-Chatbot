from pathlib import Path
from langchain_community.vectorstores import Chroma

from src.ingestion.chunk_text import chunk_documents
from src.ingestion.load_pdfs import load_user_pdfs

def build_user_vectorstore(user_id: str):
    docs = load_user_pdfs(user_id)
    chunks = chunk_documents(docs)

    embedding_model = ()

    persist_dir = Path(f"db/chroma/{user_id}")
    persist_dir.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(persist_dir)
    )

    vectorstore.persist()
    return vectorstore