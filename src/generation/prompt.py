def build_prompt(context: str, question: str) -> str:
    return f"""
You are a strict AI assistant.

Rules:
- Answer ONLY using the provided context.
- Be concise.
- Do NOT add external knowledge.
- If the answer is not present, say exactly:
  "Not found in the provided documents."

Context: 
{context}

Question: 
{question}
""".strip()