from transformers import pipeline

def load_llm():
    try:
        return pipeline(
            task="text2text-generation",
            model="google/flan-t5-large",
            max_new_tokens=256,
            truncation=True,
            model_kwargs={"max_length": 512, "local_files_only": True}
        )
    except Exception:
        return pipeline(
            task="text2text-generation",
            model="google/flan-t5-large",
            max_new_tokens=256,
            truncation=True,
            model_kwargs={"max_length": 512}
        )
