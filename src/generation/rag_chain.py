from .prompt import build_prompt

def run_rag(llm, retriever, query: str):
    docs = retriever.invoke(query)

    if not docs:
        return {
            "answer": "Not found in the provided documents.",
            "sources": []
        }

    context_blocks = []
    sources = []

    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        context_blocks.append(f"[Source {i+1}: {source}]\n{doc.page_content}")
        sources.append(source)
    
    context = "\n\n".join(context_blocks)
    prompt = build_prompt(context=context, question=query)

    output = llm(prompt)
    answer = output[0]["generated_text"].strip()

    if "Not found in the provided documents" in answer:
        return {
            "answer": "Not found in the provided documents.",
            "sources": []
        }

    return {
        "answer": answer,
        "sources": list(set(sources))
    }
