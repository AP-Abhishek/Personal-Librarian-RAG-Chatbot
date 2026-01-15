import logging
from .prompt import build_prompt
from src.utils import clean_answer, extract_pdf_name, extract_pdf_page, extract_snippet, compute_confidence

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def run_rag(llm, retriever, query: str):
    docs = retriever.invoke(query)

    if not docs:
        logging.info(f"query='{query}' | result=REFUSED | reason=no_docs")
        return {
            "answer": "Not found in the provided documents.",
            "sources": [],
            "confidence": 0.0,
            "refusal_reason": "No relevant documents found."
        }

    MIN_DOCS_REQUIRED = 1

    if len(docs) < MIN_DOCS_REQUIRED:
        logging.info(f"query='{query}' | result=REFUSED | reason=insufficient_docs")
        return {
            "answer": None,
            "sources": [],
            "confidence": 0.0,
            "refusal_reason": "Insufficient documents evidence."
        }
    
    context_blocks = []
    sources = []

    for i, doc in enumerate(docs):
        metadata = doc.metadata or {}

        pdf_name = extract_pdf_name(metadata.get("source"))
        pdf_page = extract_pdf_page(metadata)
        snippet = extract_snippet(doc.page_content)

        context_blocks.append(f"[Source {i+1}: {pdf_name} | Page: {pdf_page}]\n{doc.page_content}")
        sources.append({
            "pdf": pdf_name,
            "page": pdf_page,
            "snippet": snippet
        })
    
    confidence = compute_confidence(sources)

    if confidence < 0.35:
        logging.info(f"query='{query}' | result=REFUSED | reason=low_confidence ({confidence})")
        return {
            "answer": None,
            "sources": [],
            "confidence": confidence,
            "refusal_reason": "Retrieval evidence was too weak to answer reliably."
        }
    
    context = "\n\n".join(context_blocks)
    prompt = build_prompt(context=context, question=query)

    output = llm(prompt)
    raw_answer = output[0]["generated_text"].strip()
    answer = clean_answer(raw_answer)

    logging.info(f"query='{query}' | result=ANSWERED | sources={len(sources)}")

    return {
        "answer": answer,
        "sources": list(set(sources)),
        "confidence": confidence
    }
