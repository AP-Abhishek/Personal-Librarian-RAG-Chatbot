from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pathlib import Path
from typing import List

def load_user_pdfs(user_id: str) -> List[Document]:
    pdf_dir = Path(f"data/uploads/{user_id}/pdfs")
    documents: List[Document] = []

    if not pdf_dir.exists():
        raise FileNotFoundError(f"User {user_id} has no PDFs uploaded.")

    for pdf_file in pdf_dir.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        for doc in docs:
            doc.metadata.update({
                "source": str(pdf_file),
                "file_name": pdf_file.name,
                "user_id": user_id
            })
            documents.append(doc)
    
    return documents