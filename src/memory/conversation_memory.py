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