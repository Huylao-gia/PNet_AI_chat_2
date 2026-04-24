from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from core.config import settings
from api.routes import router as chat_router

# Import các Plugins
from plugins.vectordb_chroma import ChromaDBPlugin
from plugins.llm_openai import OpenAIPlugin
from plugins.memory_local import LocalMemoryPlugin

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print(f"🚀 KHỞI ĐỘNG HỆ THỐNG: {settings.PROJECT_NAME}")
    print("=" * 50)
    
    # DEPENDENCY INJECTION: Khởi tạo các Plugins và lưu vào App State
    app.state.vectordb = ChromaDBPlugin(
        db_path=settings.CHROMA_DB_PATH,
        collection_name=settings.COLLECTION_NAME,
        model_name=settings.EMBEDDING_MODEL
    )
    
    # Cơ chế Swap LLM dễ dàng sau này
    if settings.ACTIVE_LLM == "openai":
        app.state.llm = OpenAIPlugin(api_key=settings.OPENAI_API_KEY)
    else:
        # Chỗ này sau sẽ khởi tạo VLLMPlugin
        print("[CẢNH BÁO] Chưa implement VLLM Plugin, đang fallback lỗi...")
        pass 
        
    app.state.memory = LocalMemoryPlugin()
    
    print("✅ Hệ thống đã sẵn sàng xử lý Request!")
    print("=" * 50)
    
    yield 
    
    print("🛑 Đang tắt hệ thống, dọn dẹp bộ nhớ...")
    app.state.vectordb = None
    app.state.llm = None
    app.state.memory = None

# Khởi tạo App
app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)

    
    
# RUNNING and CHECKING GUIDE
# cd backend
# python main.py
# *(Bạn sẽ thấy Log báo nạp DB và Warm-up thành công)*

# **Bước 2:** Bật thêm một cửa sổ Terminal mới (Giả lập Website gọi tới) và chạy lệnh `curl` này:

# ```bash
# curl -N -X POST http://localhost:8080/api/chat \
#      -H "Content-Type: application/json" \
#      -d '{"session_id": "test_001", "message": "Chó nhà tôi bị nôn mửa, phải làm sao?"}'

# **Kết quả kỳ vọng:**
# Nếu phía dưới con vLLM của bạn đang chạy (ở cổng 8000), bạn sẽ thấy trên màn hình Terminal của cURL in ra từng cụm `data: {"content": "..."}` giống hệt như cách ChatGPT hiện chữ lên trình duyệt!

# Nếu bạn chưa bật vLLM, API sẽ báo lỗi Connection Refused (do chưa có AI). Vậy bạn đã cài đặt chạy thử vLLM Engine với file `.gguf` bao giờ chưa? Nếu chưa, Phase tiếp theo tôi sẽ hướng dẫn bạn viết kịch bản **Docker Compose (Phase 4)** để ép vLLM chạy ngầm nhé.
