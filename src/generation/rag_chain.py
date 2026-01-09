from .prompt import build_prompt

def run_rag(llm, retriever, query: str):
    docs = retriever.invoke(query)

    if not docs:
        return {
            "answer": "Not found in the provided documents.",
            "sources": []
        }
    
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = build_prompt(context=context, question=query)

    result = llm(prompt)[0]["generated_text"]

    sources = list({
        doc.metadata.get("source", "unknown")
        for doc in docs
    })

    return {
        "answer": result,
        "sources": sources
    }
