from transformers import pipeline

def load_llm():
    return pipeline(
        task="text2text-generation",
        model="google/flan-t5-large",
        max_new_tokens=256
    )