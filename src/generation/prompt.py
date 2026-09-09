def build_prompt(context: str, question: str) -> str:
    return f"""Answer the question based on the context below. Keep the answer complete, accurate, and concise.

Context:
{context}

Question:
{question}

Answer:""".strip()