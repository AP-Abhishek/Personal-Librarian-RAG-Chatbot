def build_prompt(context: str, question: str) -> str:
    return f"""Context:
{context}

Question: {question}

Based on the context above, answer the question concisely. If the answer is not in the context, reply "Not found in the provided documents."
Answer:""".strip()