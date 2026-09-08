import json
from pathlib import Path
from collections import deque
from typing import List

class ConversationMemory:
    def __init__(self, max_size: int = 10, persist_path: Path | None = None):
        self.max_size = max_size
        self.history = deque(maxlen=max_size)
        self.persist_path = persist_path

        if self.persist_path:
            self._load()
        
    def _load(self):
        if self.persist_path.exists():
            try:
                data = json.load(self.persist_path.read_text(encoding="utf-8"))
                for q in data.get("history", []):
                    self.history.append(q)
            except Exception:
                self.history.clear()
    
    def _save(self):
        if not self.persist_path:
            return
        
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "history": list(self.history)
        }
        self.persist_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8"
        )
    
    def add_user_query(self, query: str):
        self.history.append(query)
        self._save()
    
    def get_history(self) -> List[str]:
        return list(self.history)
    
    def clear(self):
        self.history.clear()
        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink(missing_ok=True)
    
    def get_last_meaningful_query(self) -> str | None:
        if not self.history:
            return None
        
        for q in reversed(self.history):
            if not self.is_vague(q):
                return q
        
        return None
    
    @staticmethod
    def is_vague(query: str) -> bool:
        vague_phrases = {
            "explain more",
            "tell me more",
            "more details",
            "what about that",
            "what about this",
            "and this",
            "continue",
            "go on",
            "elaborate",
            "more",
            "why",
            "how",
            "what else",
            "and then"
        }
        
        q = query.strip().lower()
        return q in vague_phrases or len(q) < 3
