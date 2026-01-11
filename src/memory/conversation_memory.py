from collections import deque
from typing import List

class ConversationMemory:
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.history = deque(maxlen=max_size)
    
    def add_user_query(self, query: str):
        self.history.append(query)
    
    def get_history(self) -> List[str]:
        return list(self.history)
    
    def clear(self):
        self.history.clear()
    
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
            "go on"
        }
        q = query.lower()
        return q in vague_phrases or len(q.split()) <= 2