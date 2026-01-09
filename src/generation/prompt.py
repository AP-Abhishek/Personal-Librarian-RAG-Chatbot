def build_prompt(context: str, question: str) -> str:
    return f"""
You are a strict AI assistant.

Answer the question based on the given context ONLY.
If the answer is not present in the context, say:
"Not found in the provided documents."

Context: 
{context}

Question: 
{question}
""".strip()