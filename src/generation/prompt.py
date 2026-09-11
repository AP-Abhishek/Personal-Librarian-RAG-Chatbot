def build_prompt(context: str, question: str) -> str:
    return f"question: {question} context: {context}".strip()