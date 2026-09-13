def build_prompt(context: str, question: str) -> str:
    q_lower = question.lower()
    if any(k in q_lower for k in ["elaborate", "detail", "comprehensive", "explain in detail", "deep dive"]):
        instruction = "Provide a detailed and thorough answer based on the provided context."
    elif any(k in q_lower for k in ["define", "what is", "briefly", "short"]):
        instruction = "Provide a direct and clear answer based on the provided context."
    else:
        instruction = "Answer the question accurately based on the provided context."

    return f"{instruction}\nquestion: {question}\ncontext: {context}".strip()