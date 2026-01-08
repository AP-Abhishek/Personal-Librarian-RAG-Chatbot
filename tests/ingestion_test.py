from src.ingestion.load_pdfs import load_user_pdfs
from src.ingestion.chunk_text import chuck_documents

docs = load_user_pdfs("user_001")
chunks = chuck_documents(docs)

print(f"Total chunks: {len(chunks)}")
print(chunks[0].page_content)
print(chunks[0].metadata)