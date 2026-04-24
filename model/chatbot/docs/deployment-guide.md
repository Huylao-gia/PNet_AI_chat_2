
# KẾ HOẠCH CHI TIẾT: ĐÓNG GÓI, TRIỂN KHAI & KIỂM THỬ

## 1. Cấu trúc cây thư mục Final (Cho Triển khai)

Đây là cấu trúc thư mục thực tế trên Server (VPS) để đảm bảo mọi thứ chạy mượt mà qua Docker Compose.

```
/opt/pet_rag_server/
├── data/
│   ├── models/
│   │   ├── model.gguf
│   │   └── model-config/
│   └── vector_db/         # Thư mục sinh ra từ script Ingest (ChromaDB)
├── backend/
│   ├── api/, core/, services/ ... # Mã nguồn Python
│   ├── requirements.txt   # fastapi, uvicorn, chromadb, openai, sentence-transformers
│   └── Dockerfile         # Dockerfile cho Backend
└── docker-compose.yml     # Orchestration File

```

## 2. Dockerfile cho Backend

Nên dùng image base là `python:3.10-slim` để giảm dung lượng.

```
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .

# Cài đặt các thư viện cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code
COPY . .

# Expose cổng 8080 cho frontend gọi
EXPOSE 8080

# Lệnh khởi chạy FastAPI bằng Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]

```

## 3. File docker-compose.yml (Trái tim của hệ thống)

File này sẽ nối mạng nội bộ giữa backend và vLLM engine, đồng thời cấu hình GPU.

```
version: '3.8'

services:
  llm_engine:
    image: vllm/vllm-openai:latest
    container_name: vllm_engine
    restart: always
    ports:
      # Không public cổng 8000 ra ngoài host để bảo mật, chỉ dùng mạng nội bộ
      - "127.0.0.1:8000:8000"
    volumes:
      - ./data/models:/app/models
    command: >
      --model /app/models/model.gguf
      --tokenizer /app/models/model-config
      --served-model-name pet-chat-model
      --max-model-len 4096
      --tensor-parallel-size 1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    networks:
      - rag_network

  api_backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: rag_api_backend
    restart: always
    ports:
      - "8080:8080" # Mở cổng này ra cho NGINX/Website gọi vào
    environment:
      - VLLM_API_URL=http://llm_engine:8000/v1
      - CHROMA_DB_PATH=/app/data/vector_db
    volumes:
      - ./data/vector_db:/app/data/vector_db # Mount DB vào backend
    depends_on:
      - llm_engine
    networks:
      - rag_network

networks:
  rag_network:
    driver: bridge

```

## 4. Chiến lược Kiểm thử (Testing Strategy)

-   **Unit Test (Trong quá trình code):** Viết pytest cho hàm `rag_engine.build_prompt` để đảm bảo Context và History không ghép đè lên nhau gây lỗi độ dài.
    
-   **Integration Test:** Đảm bảo Backend gọi được vLLM qua `http://llm_engine:8000`.
    
-   **Streaming Test (cURL):**
    
    ```
    # Test trực tiếp lên cổng 8080 của server
    curl -N -X POST http://localhost:8080/api/chat \
         -H "Content-Type: application/json" \
         -d '{"session_id": "user123", "message": "Chó nhà tôi bị bỏ ăn"}'
    
    ```
    
    _Kỳ vọng:_ Dữ liệu trả về liên tục dưới dạng:
    
    `data: Chào`
    
    `data: bạn,`
    
    `data: theo`
    
    `data: tài`
    
    `data: liệu...`
