from typing import List, Dict
from core.interfaces import BaseMemory

class LocalMemoryPlugin(BaseMemory):
    """Implement bộ nhớ lưu trực tiếp trên RAM của Server (Dictionary)."""
    
    def __init__(self):
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        self._max_history_length = 20 # Giới hạn chống tràn RAM

    def get_history(self, session_id: str, max_msgs: int = 6) -> List[Dict[str, str]]:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        # Chỉ lấy N tin nhắn gần nhất
        return self._sessions[session_id][-max_msgs:]

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self._sessions:
            self._sessions[session_id] = []
            
        self._sessions[session_id].append({"role": role, "content": content})
        
        # Cắt tỉa nếu lịch sử quá dài
        if len(self._sessions[session_id]) > self._max_history_length:
            self._sessions[session_id] = self._sessions[session_id][-self._max_history_length:]
