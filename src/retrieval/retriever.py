from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever

def load_user_vectorstore(user_id: str) -> Chroma:
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    persist_dir = Path(f"db/chroma/{user_id}")
    if not persist_dir.exists():
        raise FileNotFoundError(f"User {user_id} has no vectorstore.")

    vectorstore = Chroma(
        collection_name=f"{user_id}_pdf_collection",
        persist_directory=str(persist_dir),
        embedding_function=embedding_model
    )

    return vectorstore

def get_retriever(
    vectorstore: Chroma,
    search_type: str = "similarity",
    k: int = 4
) -> BaseRetriever:
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )