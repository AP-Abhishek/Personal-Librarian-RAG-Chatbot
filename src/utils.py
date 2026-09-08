import os

def clean_answer(text: str, max_chars: int = 800) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text.strip()
    
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        return truncated[:last_space].strip() + "..."
    return truncated.strip() + "..."

def extract_pdf_name(source: str) -> str:
    if not source:
        return "unknown"
    return os.path.basename(str(source))

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

def format_export_text(data: dict) -> str:
    lines = [
        f"Question:\n{data['question']}\n",
        f"Answer:\n{data['answer']}\n",
        f"Confidence:\n{data['confidence']}\n",
        "Sources:"
    ]

    for src in data["sources"]:
        lines.append(f" - {src['pdf']} | Page {src['page']}\n {src['snippet']}")
    
    return "\n".join(lines)

def format_export_markdown(data: dict) -> str:
    md = [
        f"## Question\n\n{data['question']}\n",
        f"## Answer\n\n{data['answer']}\n",
        f"**Confidence:** `{data['confidence']}`\n",
        "## Sources"
    ]

    for src in data["sources"]:
        md.append(f"- **{src['pdf']}**, page {src['page']}\n\n  > {src['snippet']}")
    
    return "\n".join(md)
