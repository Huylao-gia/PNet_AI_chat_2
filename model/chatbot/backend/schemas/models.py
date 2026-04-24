from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    """
    Validate payload (dữ liệu đầu vào) gửi lên từ Frontend cho API /api/chat.
    Đảm bảo mọi request đều có đủ session_id và câu hỏi.
    """
    session_id: str = Field(..., description="ID của phiên chat để duy trì ngữ cảnh hội thoại")
    message: str = Field(..., description="Câu hỏi y khoa từ người dùng")
    top_k: int = Field(default=3, description="Số lượng tài liệu tham chiếu muốn lấy từ VectorDB")

class ContextDocument(BaseModel):
    """
    Chuẩn hóa cấu trúc dữ liệu của tài liệu được trích xuất từ các VectorDB Plugins.
    Giúp các tầng Service (RAG Engine) đọc dữ liệu an toàn mà không sợ sai key của Dictionary.
    """
    page: str = Field(..., description="Trang tài liệu tham chiếu (VD: '12', 'N/A')")
    content: str = Field(..., description="Nội dung đoạn văn bản trích xuất")
    confidence_score: float = Field(..., description="Điểm tin cậy quy đổi ra % (VD: 85.5)")
