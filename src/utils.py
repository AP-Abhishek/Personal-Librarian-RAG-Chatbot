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

def compute_confidence(sources: list) -> float:
    if not sources:
        return 0.0
    
    num_chunks = len(sources)
    unique_pages = len({s["page"] for s in sources if s["page"] != "unknown"})
    avg_snippet_len = sum(len(s["snippet"]) for s in sources) / num_chunks

    chunk_score = min(num_chunks/4, 1.0)
    page_score = min(unique_pages/3, 1.0)
    content_score = 1.0 if avg_snippet_len >= 120 else 0.6

    confidence = (
        0.5 * chunk_score +
        0.3 * page_score +
        0.2 * content_score
    )

    return round(confidence, 2)
