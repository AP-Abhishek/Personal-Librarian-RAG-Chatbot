import os

def clean_answer(text: str) -> str:
    text = " ".join(text.split()).strip()
    for prefix in ["Detailed Answer:", "Answer:", "The answer is:", "Based on the context,"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text.strip()

def normalize_query(query: str) -> str:
    if not query:
        return ""
    
    q = query.strip()
    
    contractions = {
        "whats": "what is",
        "what's": "what is",
        "where's": "where is",
        "how's": "how is",
        "who's": "who is",
        "can't": "cannot",
        "don't": "do not",
        "doesn't": "does not",
        "won't": "will not",
    }
    
    words = q.split()
    normalized_words = []
    
    for word in words:
        w_clean = word.lower()
        if w_clean in contractions:
            normalized_words.append(contractions[w_clean])
        elif "%" in word:
            normalized_words.append(word.replace("%", " percentage"))
        else:
            normalized_words.append(word)
            
    result = " ".join(normalized_words)
    return " ".join(result.split())

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

def format_chat_export_text(chat_history: list) -> str:
    if not chat_history:
        return "No conversation history."
    
    lines = ["=== Personal Librarian Chat History ===\n"]
    for msg in chat_history:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")
        lines.append(f"[{role}]: {content}")
        
        sources = msg.get("sources", [])
        if sources and msg.get("role") == "assistant":
            lines.append("  Sources & References:")
            for src in sources:
                if isinstance(src, dict):
                    pdf = src.get("pdf", "Document")
                    page = src.get("page", "1")
                    snippet = src.get("snippet", "")
                    lines.append(f"   - {pdf} (Page {page}): \"{snippet}\"")
                else:
                    lines.append(f"   - {src}")
        lines.append("")
    
    return "\n".join(lines)

def format_chat_export_markdown(chat_history: list) -> str:
    if not chat_history:
        return "*No conversation history.*"
    
    md = ["# 📚 Personal Librarian - Conversation Export\n"]
    for msg in chat_history:
        role = "👤 **User**" if msg.get("role") == "user" else "🤖 **Assistant**"
        content = msg.get("content", "")
        md.append(f"### {role}\n{content}\n")
        
        sources = msg.get("sources", [])
        if sources and msg.get("role") == "assistant":
            md.append("**References & Sources:**")
            for src in sources:
                if isinstance(src, dict):
                    pdf = src.get("pdf", "Document")
                    page = src.get("page", "1")
                    snippet = src.get("snippet", "")
                    md.append(f"- 📄 **{pdf}** (Page {page})\n  > *\"{snippet}\"*")
                else:
                    md.append(f"- {src}")
            md.append("")
        md.append("---\n")
    
    return "\n".join(md)
