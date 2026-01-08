from src.ingestion.load_pdfs import load_user_pdfs
from src.ingestion.chunk_text import chunk_documents

def run_test():
    docs = load_user_pdfs("user_001")
    chunks = chunk_documents(docs)

    print(f"Loaded Pages: {len(docs)}")
    print(f"Generated Chunks: {len(chunks)}")

    sample = chunks[0]
    print(f"Sample Content: {sample.page_content[:500]}")
    print(f"Sample Metadata: {sample.metadata}")

if __name__ == "__main__":
    run_test()