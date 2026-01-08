from langchain.document_loaders import PyPDFLoader, Document
from pathlib import Path
from typing import List

def load_user_pdfs(user_id: str) -> List[Document]:
    pdf_dir = Path(f"data/uploads/{user_id}/pdfs")
    documents = []

    for pdf_file in pdf_dir.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = pdf_file.name
            documents.append(doc)
    
    return documents