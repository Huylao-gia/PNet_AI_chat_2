from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncGenerator

class BaseVectorDB(ABC):
    """Giao diện chuẩn cho mọi loại Vector Database."""
    
    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Tìm kiếm ngữ nghĩa.
        Yêu cầu trả về list các dict, mỗi dict chứa ít nhất:
        - 'content': str (Nội dung văn bản)
        - 'page': str/int (Trang tài liệu)
        - 'score': float (Điểm tin cậy / Độ tương đồng)
        """
        pass

class BaseLLM(ABC):
    """Giao diện chuẩn cho mọi Mô hình Ngôn ngữ (OpenAI, vLLM, HuggingFace...)."""
    
    @abstractmethod
    async def stream_chat(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """
        Nhận vào danh sách tin nhắn (role, content) và trả về luồng string bất đồng bộ.
        """
        pass

class BaseMemory(ABC):
    """Giao diện chuẩn cho hệ thống lưu trữ lịch sử hội thoại."""
    
    @abstractmethod
    def get_history(self, session_id: str, max_msgs: int = 6) -> List[Dict[str, str]]:
        """Lấy lịch sử tin nhắn của một session."""
        pass
        
    @abstractmethod
    def add_message(self, session_id: str, role: str, content: str):
        """Thêm một tin nhắn mới vào lịch sử."""
        pass
