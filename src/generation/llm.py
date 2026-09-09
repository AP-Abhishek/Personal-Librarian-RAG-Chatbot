from transformers import pipeline

def load_llm():
    gen_kwargs = {
        "max_new_tokens": 160,
        "num_beams": 2,
        "early_stopping": True,
        "no_repeat_ngram_size": 3,
        "truncation": True,
    }
    try:
        return pipeline(
            task="text2text-generation",
            model="google/flan-t5-large",
            model_kwargs={"local_files_only": True},
            **gen_kwargs
        )
    except Exception:
        return pipeline(
            task="text2text-generation",
            model="google/flan-t5-large",
            **gen_kwargs
        )

