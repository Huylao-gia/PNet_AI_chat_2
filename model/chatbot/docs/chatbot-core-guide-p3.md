# PHASE 4: ĐÓNG GÓI DOCKER & TRIỂN KHAI HỆ THỐNG

## 1. Chuẩn hóa cấu trúc thư mục gốc (Root Directory)

Trước khi viết file Docker, hãy đảm bảo thư mục gốc dự án của bạn (thư mục `chatbot/`) đang có cấu trúc tương tự thế này:

```
chatbot/
├── backend/                        # Code FastAPI chúng ta vừa viết ở Phase 3
│   ├── api/, core/, schemas/, services/
│   ├── main.py
│   └── requirements.txt
├── vectordb-processing/            
│   └── chroma_db/                  # Thư mục Database bạn đã sinh ra ở Phase 2
├── model.gguf                      # File trọng số AI đã Finetune
├── model-config/                   # Thư mục chứa tokenizer.json, config.json...
├── docker-compose.yml              # (Sắp tạo) Trái tim điều phối hệ thống
└── backend.Dockerfile              # (Sắp tạo) Kịch bản đóng gói Backend

```

## 2. Tạo kịch bản đóng gói Backend (`backend.Dockerfile`)

Bạn hãy tạo một file tên là `backend.Dockerfile` nằm ở thư mục gốc `chatbot/`. File này hướng dẫn Docker cách bọc mã nguồn Python của bạn thành một "chiếc hộp" (Container) có thể tự chạy.

```
# filepath: backend.Dockerfile
# Sử dụng phiên bản Python 3.10 mỏng nhẹ để tối ưu dung lượng
FROM python:3.10-slim

# Thiết lập thư mục làm việc trong Container
WORKDIR /app

# Khắc phục lỗi thiếu thư viện hệ thống khi cài đặt một số package Python
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt thư viện Python trước (Tận dụng cache của Docker để build nhanh hơn)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn Backend vào Container
COPY backend/ /app/

# Khai báo cổng 8080 cho API
EXPOSE 8080

# Lệnh khởi chạy server bằng Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]

```

## 3. Trái tim điều phối hệ thống (`docker-compose.yml`)

Đây là file quan trọng nhất. Nó sẽ gọi 2 Container chạy lên cùng lúc:

1.  **vLLM Engine:** Chạy trực tiếp từ image chính chủ của vLLM, gánh file `model.gguf` trên GPU.
    
2.  **API Backend:** Chạy từ file `backend.Dockerfile` ở trên.
    

Tạo file `docker-compose.yml` ở thư mục gốc `chatbot/`:

```
# filepath: docker-compose.yml
version: '3.8'

services:
  # -----------------------------------------------------
  # 1. MICROSERVICE: LLM ENGINE (Chạy vLLM trên GPU)
  # -----------------------------------------------------
  llm_engine:
    image: vllm/vllm-openai:latest
    container_name: pet_vllm_engine
    restart: unless-stopped
    # CHÚ Ý: Chỉ expose cổng nội bộ 8000, KHÔNG public ra host để tránh bị gọi lén API
    expose:
      - "8000"
    # Ánh xạ model và config vào trong container
    volumes:
      - ./model.gguf:/app/models/model.gguf:ro
      - ./model-config:/app/models/model-config:ro
    # Lệnh khởi chạy vLLM (Chuẩn xác cho file GGUF)
    command: >
      --model /app/models/model.gguf
      --tokenizer /app/models/model-config
      --served-model-name pet-chat-model
      --max-model-len 4096
      --tensor-parallel-size 1
      --gpu-memory-utilization 0.85
      --port 8000
    # Yêu cầu kích hoạt GPU của máy chủ
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - rag_network

  # -----------------------------------------------------
  # 2. MICROSERVICE: BACKEND API (FastAPI + ChromaDB)
  # -----------------------------------------------------
  api_backend:
    build:
      context: .
      dockerfile: backend.Dockerfile
    container_name: pet_rag_backend
    restart: unless-stopped
    # Public cổng 8080 ra ngoài cho Website/Trình duyệt truy cập
    ports:
      - "8080:8080"
    # Ánh xạ VectorDB vào trong container
    volumes:
      - ./vectordb-processing/chroma_db:/app/data/vector_db
    # Biến môi trường báo cho Backend biết cách kết nối với các thành phần khác
    environment:
      - VLLM_API_URL=http://llm_engine:8000/v1
      - CHROMA_DB_PATH=/app/data/vector_db
      - COLLECTION_NAME=pet_medical_docs
      - EMBEDDING_MODEL=keepitreal/vietnamese-sbert
    depends_on:
      - llm_engine
    networks:
      - rag_network

# Định nghĩa mạng nội bộ để 2 container gọi nhau qua tên (llm_engine, api_backend)
networks:
  rag_network:
    driver: bridge

```

## 4. Quy trình Khởi chạy & Kiểm thử

### Bước 4.1: Cài đặt Nvidia Container Toolkit (Nếu chưa có trên VPS)

Để Docker nhận diện được Card Màn Hình (GPU), hệ điều hành Ubuntu của bạn bắt buộc phải có `nvidia-container-toolkit`.

```
# Kiểm tra xem máy đã nhận GPU chưa:
nvidia-smi

# Nếu chưa có toolkit, hãy cài đặt theo docs của NVIDIA.

```

### Bước 4.2: Khởi động toàn bộ hệ thống

Mở Terminal tại thư mục `chatbot/`, chạy lệnh sau (Quá trình build lần đầu có thể mất 3-5 phút):

```
docker-compose up -d --build

```

Cờ `-d` giúp hệ thống chạy ngầm.

### Bước 4.3: Theo dõi Log (Rất quan trọng)

Bạn cần xem hệ thống đang load model đến đâu. File GGUF nạp lên VRAM thường mất một lúc.

```
# Xem log của vLLM Engine
docker logs -f pet_vllm_engine

# Xem log của Backend (Đợi vLLM báo "Uvicorn running on [http://0.0.0.0:8000](http://0.0.0.0:8000)" thì backend mới gọi được)
docker logs -f pet_rag_backend

```

### Bước 4.4: Kiểm thử API từ bên ngoài

Mở một Terminal khác hoặc dùng phần mềm Postman/cURL để đóng vai Website gửi request đến hệ thống của bạn:

```
curl -N -X POST http://localhost:8080/api/chat \
     -H "Content-Type: application/json" \
     -d '{
           "session_id": "test_user_001", 
           "message": "Chó nhà em có biểu hiện co giật, sùi bọt mép. Liệu có phải bị dại không?",
           "top_k": 3
         }'

```

**Dấu hiệu thành công rực rỡ:**

Bạn sẽ thấy trên màn hình Terminal in ra lần lượt từng chữ:

`data: {"content": "Chào "}`

`data: {"content": "bạn, "}`

`data: {"content": "dựa "}`

`data: {"content": "vào "}`...

Và kết thúc bằng dòng:

`data: [DONE]`
