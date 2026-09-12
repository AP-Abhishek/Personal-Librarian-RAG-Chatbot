from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

def load_llm():
    model_name = "google/flan-t5-large"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=False,
            truncation=True
        )
    except Exception:
        return pipeline(
            "text2text-generation",
            model=model_name,
            max_new_tokens=256,
            do_sample=False,
            truncation=True
        )

