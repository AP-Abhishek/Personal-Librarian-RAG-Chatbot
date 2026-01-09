def build_prompt(context: str, question: str) -> str:
    return f"""
You are a strict AI assistant.

Answer using the provided context.
You may rephrase or summarize.
If the answer is truly missing, say:
"Not found in the provided documents."

Context: 
{context}

Question: 
{question}
""".strip()