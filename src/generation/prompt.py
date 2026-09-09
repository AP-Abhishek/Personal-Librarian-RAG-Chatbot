def build_prompt(context: str, question: str, chat_history: str = "") -> str:
    history_prefix = f"Previous Conversation:\n{chat_history}\n\n" if chat_history else ""
    return f"""{history_prefix}Answer the question based on the context below. List all relevant projects, items, or details completely and concisely.

Context:
{context}

Question:
{question}

Answer:""".strip()