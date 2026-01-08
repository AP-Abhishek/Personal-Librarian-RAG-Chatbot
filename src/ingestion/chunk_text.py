from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.document_loaders import Document

def chuck_documents(documents: List[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chuck_overlap=100
    )

    chunks = splitter.split_documents(documents)
    return chunks