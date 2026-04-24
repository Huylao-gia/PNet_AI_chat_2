
# IMPLEMENT CHI TIẾT TẦNG DỊCH VỤ (SERVICES) & API

Hãy lần lượt mở các file trống trong dự án của bạn và dán các đoạn code tương ứng dưới đây vào.

## 1. Dịch Vụ Lưu Trữ Dữ Liệu (`services/vector_store.py`)

File này chịu trách nhiệm "chạm" vào bộ nhớ RAM (đã được nạp ở `main.py`) để lấy model và DB ra tìm kiếm siêu tốc.

```
# filepath: backend/services/vector_store.py
from typing import List
from fastapi import Request
from schemas.models import ContextDocument

async def search_context(query: str, request: Request, top_k: int = 3) -> List[ContextDocument]:
    """Tìm kiếm ngữ nghĩa trong VectorDB đã được nạp sẵn trên RAM."""
    embedder = request.app.state.embedder
    collection = request.app.state.collection
    
    # 1. Chuyển đổi câu hỏi thành Vector
    query_vector = embedder.encode([query], show_progress_bar=False).tolist()
    
    # 2. Truy vấn
    results = collection.query(
        query_embeddings=query_vector,
        n_results=top_k
    )
    
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    contexts = []
    for i in range(len(documents)):
        # Chuyển đổi khoảng cách Cosine thành % tự tin
        confidence = round((1 - distances[i]) * 100, 1)
        contexts.append(ContextDocument(
            page=str(metadatas[i].get("page", "N/A")),
            content=documents[i],
            confidence_score=confidence
        ))
        
    return contexts

```

## 2. Quản Lý Lịch Sử Chat (`services/memory.py`)

Mô phỏng bộ nhớ lưu hội thoại (Session Memory). Sử dụng Sliding Window (chỉ giữ lại N tin nhắn gần nhất) để tránh làm tràn Context Window của LLM.

```
# filepath: backend/services/memory.py
from typing import List, Dict

# Giả lập In-memory DB. Trong thực tế (Production quy mô lớn), bạn nên thay bằng Redis.
SESSIONS: Dict[str, List[Dict[str, str]]] = {}

def get_chat_history(session_id: str, max_messages: int = 6) -> List[Dict[str, str]]:
    """Lấy lịch sử chat. Mặc định lấy 6 tin gần nhất (3 lượt hỏi - đáp)."""
    if session_id not in SESSIONS:
        SESSIONS[session_id] = []
    return SESSIONS[session_id][-max_messages:]

def add_message(session_id: str, role: str, content: str):
    """Thêm tin nhắn mới vào session. Role có thể là 'user' hoặc 'assistant'."""
    if session_id not in SESSIONS:
        SESSIONS[session_id] = []
        
    SESSIONS[session_id].append({"role": role, "content": content})
    
    # Garbage Collection: Không giữ quá 20 tin nhắn mỗi user để tiết kiệm RAM server
    if len(SESSIONS[session_id]) > 20:
        SESSIONS[session_id] = SESSIONS[session_id][-20:]

```

## 3. Lõi RAG Ghép Prompt (`services/rag_engine.py`)

Biến đổi Tài liệu từ DB + Lịch sử hội thoại thành Format mà AI (vLLM) có thể hiểu được (như format của ChatGPT).

```
# filepath: backend/services/rag_engine.py
from typing import List
from schemas.models import ContextDocument

def build_prompt_messages(contexts: List[ContextDocument], history: List[Dict[str, str]]) -> list:
    """
    Xây dựng chuẩn tin nhắn (messages array) để gửi tới LLM.
    - System prompt (chứa Context từ DB)
    - History (chứa các câu hỏi đáp cũ + câu hỏi mới nhất của user)
    """
    # 1. Định dạng tài liệu tham khảo
    context_str = ""
    if contexts:
        for idx, ctx in enumerate(contexts):
            context_str += f"[Tài liệu {idx+1} | Trang {ctx.page} | Độ chính xác {ctx.confidence_score}%]:\n{ctx.content}\n\n"
    else:
        context_str = "Không tìm thấy tài liệu tham khảo nào trong cơ sở dữ liệu."

    # 2. Xây dựng System Prompt (Ép AI phải tuân thủ nghiêm ngặt RAG)
    system_prompt = f"""Bạn là một chuyên gia Thú y AI ảo chuyên nghiệp và tận tâm.
NHIỆM VỤ CỦA BẠN: Trả lời câu hỏi của người dùng CHỈ DỰA VÀO phần "TÀI LIỆU THAM KHẢO" dưới đây.

QUY TẮC NGHIÊM NGẶT:
1. Nếu thông tin CÓ TRONG tài liệu, hãy tổng hợp và trả lời dễ hiểu, có thể liệt kê các bước nếu cần.
2. Nếu thông tin KHÔNG CÓ TRONG tài liệu, hãy nói rõ: "Dựa trên tài liệu y khoa hiện tại, tôi không tìm thấy thông tin này...", TUYỆT ĐỐI KHÔNG tự bịa đặt kiến thức (Hallucination).
3. Sử dụng tiếng Việt chuẩn xác, giọng điệu đồng cảm với người nuôi thú cưng.

TÀI LIỆU THAM KHẢO:
{context_str}
"""
    
    # 3. Lắp ráp cấu trúc cuối cùng
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history) # Đưa toàn bộ ngữ cảnh trò chuyện vào
    
    return messages

```

## 4. Client Kết Nối AI (`services/llm_client.py`)

Sử dụng thư viện `openai` nhưng trỏ API URL về thẳng con **vLLM Engine** của chúng ta.

```
# filepath: backend/services/llm_client.py
from openai import AsyncOpenAI
from core.config import settings

# Khởi tạo Async Client. Trỏ base_url tới vLLM (Ví dụ: http://localhost:8000/v1)
client = AsyncOpenAI(
    base_url=settings.VLLM_API_URL,
    api_key="EMPTY_KEY" # vLLM cục bộ không cần API Key
)

async def stream_generate(messages: list):
    """Gọi vLLM và trả về từng chữ (token) một ngay lập tức."""
    response = await client.chat.completions.create(
        model="pet-chat-model", # Tên model phải khớp với lệnh khởi chạy vLLM
        messages=messages,
        stream=True,         # Bật Streaming
        temperature=0.1,     # Nhiệt độ thấp (0.1) giúp AI không bị "ảo giác", trả lời chuẩn theo y khoa
        max_tokens=1024,
        presence_penalty=0.2 # Khuyến khích AI dùng từ đa dạng một chút
    )
    
    async for chunk in response:
        # Lấy token vừa được sinh ra (nếu có)
        if chunk.choices and chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

```

## 5. Ghép Mạch API Cuối Cùng (`api/routes.py`)

Nơi tiếp nhận Request từ Website, gọi tất cả 4 service trên và trả luồng Server-Sent Events (SSE) về cho trình duyệt.

```
# filepath: backend/api/routes.py
import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from schemas.models import ChatRequest
from services import vector_store, memory, rag_engine, llm_client

router = APIRouter()

@router.post("/api/chat")
async def chat_endpoint(request: Request, body: ChatRequest):
    session_id = body.session_id
    user_msg = body.message
    
    # 1. Lưu ngay câu hỏi của người dùng vào Session Memory
    memory.add_message(session_id, "user", user_msg)
    
    # 2. Truy xuất tài liệu từ VectorDB siêu tốc
    contexts = await vector_store.search_context(user_msg, request, top_k=body.top_k)
    
    # 3. Lấy lịch sử chat (đã bao gồm câu hỏi vừa lưu ở bước 1)
    history = memory.get_chat_history(session_id, max_messages=6)
    
    # 4. Gói ghém tất cả thành Prompt chuẩn mực
    messages = rag_engine.build_prompt_messages(contexts, history)
    
    # 5. Hàm Generator phát sự kiện SSE
    async def event_generator():
        full_ai_response = ""
        try:
            # Lặp qua từng Token được vLLM nhả ra
            async for token in llm_client.stream_generate(messages):
                full_ai_response += token
                # Đóng gói Token thành chuỗi JSON an toàn cho giao thức SSE
                payload = json.dumps({"content": token}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                
        finally:
            # 6. KHI HOÀN THÀNH (hoặc khi user tắt trình duyệt giữa chừng):
            # Lưu lại toàn bộ câu trả lời của AI vào Memory để dùng cho lượt hỏi sau
            if full_ai_response:
                memory.add_message(session_id, "assistant", full_ai_response)
            
            # Báo hiệu cho Website biết đã stream xong
            yield "data: [DONE]\n\n"

    # Trả về Response chuẩn Streaming
    return StreamingResponse(event_generator(), media_type="text/event-stream")

```
