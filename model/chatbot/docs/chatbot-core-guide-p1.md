
# HƯỚNG DẪN SETUP BASE PROJECT CHO RAG BACKEND

## 1. Tạo cấu trúc thư mục tự động

Hãy mở Terminal (trên Linux/Mac hoặc Git Bash/PowerShell trên Windows), di chuyển vào thư mục dự án `chatbot/` và chạy đoạn mã script sau để tự động tạo cây thư mục và các file trống:

```
mkdir -p backend/api backend/core backend/services backend/schemas
touch backend/api/__init__.py backend/api/routes.py
touch backend/core/__init__.py backend/core/config.py
touch backend/services/__init__.py backend/services/embedding.py backend/services/vector_store.py backend/services/rag_engine.py backend/services/llm_client.py backend/services/memory.py
touch backend/schemas/__init__.py backend/schemas/models.py
touch backend/main.py backend/requirements.txt backend/.env

```

Sau khi chạy xong, thư mục `backend/` của bạn sẽ trông như thế này:

```
backend/
├── .env                  # Biến môi trường local
├── requirements.txt      # Thư viện
├── main.py               # File chạy chính của FastAPI
├── api/
│   └── routes.py         # Khai báo các endpoint (POST /api/chat)
├── core/
│   └── config.py         # Quản lý cấu hình toàn cục
├── schemas/
│   └── models.py         # Khai báo cấu trúc dữ liệu (Input/Output validate)
└── services/
    ├── embedding.py      # Xử lý SBERT
    ├── llm_client.py     # Gọi vLLM
    ├── memory.py         # Xử lý lịch sử & Summarize
    ├── rag_engine.py     # Ghép Prompt
    └── vector_store.py   # Xử lý ChromaDB

```

## 2. File Thư viện Yêu cầu (`requirements.txt`)

Mở file `backend/requirements.txt` và dán nội dung sau. Các phiên bản đã được ghim lại để đảm bảo tính ổn định.

```
fastapi==0.109.2
uvicorn==0.27.1
pydantic==2.6.1
pydantic-settings==2.1.0
chromadb==0.4.24
sentence-transformers==2.5.1
openai==1.12.0          # Dùng OpenAI client để gọi giao thức chuẩn của vLLM
httpx==0.26.0           # Hỗ trợ gọi API bất đồng bộ (async)

```

_Chạy lệnh cài đặt: `cd backend && pip install -r requirements.txt`_

## 3. Biến môi trường (`.env`)

Mở file `backend/.env` và thêm cấu hình. File này giúp ta dễ thay đổi đường dẫn khi chuyển từ Local lên Docker mà không cần sửa code.

```
PROJECT_NAME="Pet Care RAG API"
VLLM_API_URL="http://localhost:8000/v1"
CHROMA_DB_PATH="../vectordb-processing/chroma_db"
COLLECTION_NAME="pet_medical_docs"
EMBEDDING_MODEL="keepitreal/vietnamese-sbert"
MAX_HISTORY_TOKENS=1500

```

## 4. Quản lý Cấu hình (`core/config.py`)

Sử dụng `pydantic-settings` để tự động đọc biến môi trường từ `.env` và gợi ý code (auto-complete) ở mọi nơi trong dự án.

```
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Pet Care RAG API"
    VLLM_API_URL: str = "http://localhost:8000/v1"
    CHROMA_DB_PATH: str = "../vectordb-processing/chroma_db"
    COLLECTION_NAME: str = "pet_medical_docs"
    EMBEDDING_MODEL: str = "keepitreal/vietnamese-sbert"
    MAX_HISTORY_TOKENS: int = 1500
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Khởi tạo object settings để dùng chung
settings = Settings()

```

## 5. Cấu trúc Dữ liệu API (`schemas/models.py`)

Mọi request gửi lên từ Website đều phải đi qua các Model này để FastAPI tự động kiểm tra tính hợp lệ (Validation).

```
from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="ID phiên chat của User, dùng để lưu lịch sử")
    message: str = Field(..., description="Câu hỏi y khoa từ người dùng")
    top_k: int = Field(default=3, description="Số lượng kết quả lấy từ VectorDB")

class ContextDocument(BaseModel):
    page: str
    content: str
    confidence_score: float

```

## 6. File khởi chạy hệ thống (`main.py`)

File này chịu trách nhiệm khởi động Server, nạp Model Embedding và Database lên RAM (theo pattern Lifespan), thiết lập CORS để Website có thể gọi tới, và gắn Router.

```
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from sentence_transformers import SentenceTransformer

from core.config import settings
from api.routes import router as chat_router

# Quản lý Lifespan: Load model 1 lần duy nhất khi bật server
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print(f"🚀 KHỞI ĐỘNG HỆ THỐNG: {settings.PROJECT_NAME}")
    print("=" * 50)
    
    # 1. Khởi tạo Vector DB
    print("[1/2] Đang kết nối ChromaDB...")
    client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    app.state.collection = client.get_or_create_collection(name=settings.COLLECTION_NAME)
    
    # 2. Khởi tạo Embedding Model
    print(f"[2/2] Đang nạp SBERT: {settings.EMBEDDING_MODEL} (CPU)...")
    import logging
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    app.state.embedder = SentenceTransformer(settings.EMBEDDING_MODEL, device="cpu")
    
    # Warm-up (Tránh Cold start)
    app.state.embedder.encode(["warm up"], show_progress_bar=False)
    
    print("✅ Hệ thống đã sẵn sàng xử lý Request!")
    print("=" * 50)
    
    yield # Máy chủ chạy tại đây
    
    print("🛑 Đang tắt hệ thống, giải phóng bộ nhớ...")
    app.state.collection = None
    app.state.embedder = None

# Khởi tạo App
app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

# Cấu hình CORS (Cho phép Website gọi API không bị lỗi Cross-Origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Trên production nên thay bằng URL của website
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tạm thời tạo router trống ở routes.py để tránh lỗi khi import
# Bổ sung dòng sau vào backend/api/routes.py:
# from fastapi import APIRouter
# router = APIRouter()

# Gắn các API endpoint vào app chính
app.include_router(chat_router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": settings.PROJECT_NAME}

if __name__ == "__main__":
    import uvicorn
    # Mặc định chạy trên cổng 8080
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)

```
