import re
from pathlib import Path
from typing import List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from src.embeddings.build_vectorstore import get_embedding_model

def load_user_vectorstore(user_id: str) -> Chroma:
    embedding_model = get_embedding_model()

    persist_dir = Path(f"db/chroma/{user_id}")
    if not persist_dir.exists():
        raise FileNotFoundError(f"User {user_id} has no vectorstore.")

    vectorstore = Chroma(
        collection_name=f"{user_id}_pdf_collection",
        persist_directory=str(persist_dir),
        embedding_function=embedding_model
    )

    return vectorstore

class HybridVectorRetriever(BaseRetriever):
    vectorstore: Chroma
    k: int = 2

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        candidates = self.vectorstore.similarity_search(query, k=8)
        if not candidates:
            return []

        stopwords = {
            "tell", "me", "the", "a", "an", "is", "are", "was", "were",
            "what", "whats", "what's", "where", "how", "in", "on", "at",
            "for", "to", "of", "and", "or", "about", "my", "your", "this",
            "can", "will", "show", "give", "list", "describe", "explain",
            "does", "did", "do", "find", "search", "please", "with", "from",
            "by", "has", "have", "had", "which", "when", "that", "these", "those"
        }
        raw_terms = re.findall(r'\b\w+\b', query.lower())
        keywords = [t for t in raw_terms if t not in stopwords and len(t) >= 2]

        scored_docs = []
        for idx, doc in enumerate(candidates):
            content_lower = doc.page_content.lower()
            keyword_score = 0.0
            
            for kw in keywords:
                if kw in content_lower:
                    keyword_score += 2.5 + content_lower.count(kw) * 0.5
            
            vector_score = 1.0 / (idx + 1)
            total_score = vector_score + keyword_score
            scored_docs.append((total_score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:self.k]]

def get_retriever(
    vectorstore: Chroma,
    search_type: str = "similarity",
    k: int = 4
) -> BaseRetriever:
    return HybridVectorRetriever(vectorstore=vectorstore, k=k)