
# KẾ HOẠCH CHI TIẾT: KIẾN TRÚC LOGIC BACKEND (RAG & CHATBOT)

## 1. Kiến trúc phân tầng (Layered Architecture)

Để hệ thống dễ maintain và expose API chuẩn, thư mục code backend FastAPI cần chia tầng như sau:

```
/backend/
├── api/
│   └── routes.py         # Chứa định nghĩa API POST /api/chat (SSE Streaming)
├── core/
│   └── config.py         # Chứa biến môi trường (VLLM_URL, DB_PATH, MAX_TOKENS)
├── services/
│   ├── embedding.py      # Init model SBERT, hàm chuyển text -> vector
│   ├── vector_store.py   # Kết nối ChromaDB, hàm query top-K
│   ├── rag_engine.py     # Lõi RAG: Nhận câu hỏi -> Lấy Context -> Ghép Prompt
│   ├── llm_client.py     # Gọi HTTP/OpenAI Async Client tới vLLM (Streaming)
│   └── memory.py         # Quản lý lịch sử hội thoại & Summarize
├── schemas/
│   └── models.py         # Pydantic models validate input/output
└── main.py               # Init FastAPI app, CORS, Middleware

```

## 2. Logic Duy trì hội thoại (Conversation Memory) & Summarization

-   **Vấn đề:** RAG cần gửi kèm Context (tài liệu lấy từ DB). Nếu gửi thêm toàn bộ lịch sử chat, context window (4096 tokens) sẽ nhanh chóng bị tràn.
    
-   **Giải pháp:** Sử dụng **Window Memory kết hợp Summarization**.
    

**Luồng xử lý (Memory Layer):**

1.  Backend lưu trữ lịch sử chat tạm thời vào bộ nhớ (hoặc Redis/SQLite) theo `session_id`.
    
2.  Khi User gửi câu hỏi mới, lấy ra X lượt chat gần nhất.
    
3.  **Đếm token:** Nếu tổng token lịch sử > Threshold (VD: 1500 tokens).
    
4.  **Trigger Summarize:** Kích hoạt một lời gọi LLM ẩn (Background task) để tóm tắt lịch sử đó thành 1 đoạn ngắn: _"Người dùng đang hỏi về triệu chứng nôn mửa của chó poodle, đã thử cho uống nước gừng..."_
    
5.  Xóa lịch sử cũ, thay thế bằng đoạn tóm tắt.
    

## 3. Quy trình ghép Prompt (Prompt Engineering cho RAG)

Tầng `rag_engine.py` sẽ làm nhiệm vụ này trước khi gọi LLM. Format Prompt sử dụng ChatML (do model config đã định nghĩa).

**Hệ thống RAG Prompt chuẩn:**

```
<|im_start|>system
Bạn là một trợ lý thú y AI ảo. 
Sử dụng tài liệu tham khảo dưới đây để trả lời câu hỏi của người dùng một cách chính xác.
Nếu tài liệu không chứa thông tin, hãy nói "Tôi không tìm thấy thông tin này trong cơ sở dữ liệu y khoa", KHÔNG tự bịa ra kiến thức.

TÀI LIỆU THAM KHẢO:
{retrieved_context}

TÓM TẮT LỊCH SỬ CHAT:
{chat_summary}
<|im_end|>
<|im_start|>user
{user_question}<|im_end|>
<|im_start|>assistant

```

## 4. Trả về cho Website (SSE - Server Sent Events)

FastAPI sử dụng `StreamingResponse` để truyền từng token từ `llm_client.py` về Frontend.

**Mã giả cho Controller (routes.py):**

```
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter()

@router.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    # 1. Truy xuất dữ liệu (VectorDB)
    context = vector_store.search(request.message, top_k=3)
    
    # 2. Xây dựng prompt
    prompt = rag_engine.build_prompt(request.message, context, request.session_id)
    
    # 3. Gọi vLLM dạng Stream
    async def event_generator():
        async for chunk in llm_client.stream_generate(prompt):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
        
    return StreamingResponse(event_generator(), media_type="text/event-stream")

```
