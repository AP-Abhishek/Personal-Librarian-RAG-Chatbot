import logging, datetime
from .prompt import build_prompt

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def clean_answer(text: str, max_chars: int = 600) -> str:
    text = " ".join(text.split())

    sentences = text.split(". ")
    seen = set()
    cleaned = []

    for s in sentences:
        if s not in seen:
            seen.add(s)
            cleaned.append(s)
        
    cleaned_text = ". ".join(cleaned)
    return cleaned_text[:max_chars].strip()

def run_rag(llm, retriever, query: str):
    docs = retriever.invoke(query)

    if not docs:
        logging.info(f"query='{query}' | result=REFUSED | reason=no_docs")
        return {
            "answer": "Not found in the provided documents.",
            "sources": []
        }
    
    context_blocks = []
    sources = []

    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        context_blocks.append(f"[Source {i+1}: {source } | Chunk: {chunk_id}]\n{doc.page_content}")
        sources.append(f"{source} | chunk {chunk_id}")
    
    context = "\n\n".join(context_blocks)
    prompt = build_prompt(context=context, question=query)

    logging.info(f"query='{query}' | result=ANSWERED | sources={len(sources)}")

    output = llm(prompt)
    raw_answer = output[0]["generated_text"].strip()
    answer = clean_answer(raw_answer)

    if "Not found in the provided documents" in answer:
        logging.info(f"query='{query}' | result=REFUSED | reason=llm_refusal")
        return {
            "answer": "Not found in the provided documents.",
            "sources": []
        }

    return {
        "answer": answer,
        "sources": list(set(sources))
    }
