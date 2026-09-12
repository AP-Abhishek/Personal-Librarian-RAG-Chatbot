from transformers import pipeline

def load_llm():
    try:
        return pipeline(
            task="text2text-generation",
            model="google/flan-t5-large",
            max_new_tokens=256,
            do_sample=False,
            truncation=True,
            model_kwargs={"local_files_only": True}
        )
    except Exception:
        return pipeline(
            task="text2text-generation",
            model="google/flan-t5-large",
            max_new_tokens=256,
            do_sample=False,
            truncation=True
        )

