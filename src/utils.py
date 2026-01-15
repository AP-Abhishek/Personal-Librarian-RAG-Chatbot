import os

def clean_answer(text: str, max_chars: int = 600) -> str:
    text = " ".join(text.split())

    sentences = text.split(". ")
    seen = set()
    cleaned = []

    for s in sentences:
        if s not in seen:
            seen.add(s)
            cleaned.append(s)
        
    cleaned_text = ". ".join(cleaned)
    return cleaned_text[:max_chars].strip()

def extract_pdf_name(source: str) -> str:
    if not source:
        return "unknown"
    return os.path.basename(source)

def extract_pdf_page(metadata: dict) -> str:
    for key in ("page", "page_number", "page_index"):
        if key in metadata:
            try:
                return str(int(metadata[key]) + 1)
            except Exception:
                pass
    return "unknown"

def extract_snippet(text: str, max_chars: int = 300) -> str:
    text = " ".join(text.split())
    return text[:max_chars].strip()    