
# KẾ HOẠCH CHI TIẾT: HOSTING LLM VỚI vLLM (GGUF)

## 1. Phân tích hiện trạng & Giải pháp

-   **Hiện trạng:** Bạn đang có thư mục gốc `/chatbot/`, trong đó có file `model.gguf` (đã finetune) và thư mục `model-config/` (chứa `tokenizer.json`, `tokenizer_config.json`, `config.json`).
    
-   **Thách thức:** vLLM natively hỗ trợ định dạng Safetensors/PyTorch từ HuggingFace rất tốt, nhưng với định dạng GGUF (lượng tử hóa), vLLM yêu cầu chỉ định đường dẫn file cụ thể và thư mục tokenizer riêng.
    

## 2. Cấu trúc thư mục tối ưu cho vLLM

Ta sẽ ánh xạ thư mục `/chatbot/` trên máy host vào `/app/models` trong Docker.

```
/chatbot/
├── model.gguf
└── model-config/
    ├── config.json
    ├── tokenizer.json
    └── tokenizer_config.json (Chứa chat_template ChatML)

```

## 3. Cấu hình vLLM Engine

Chúng ta sẽ sử dụng Docker Image chính thức của vLLM. Tuy nhiên, thay vì dùng command ngắn, ta cần truyền tham số chi tiết để vLLM nhận diện GGUF và Tokenizer rời.

### Lệnh khởi chạy cốt lõi (sẽ đưa vào docker-compose):

```
python3 -m vllm.entrypoints.openai.api_server \
    --model /app/models/model.gguf \
    --tokenizer /app/models/model-config \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --port 8000 \
    --served-model-name pet-chat-model

```

### Giải thích tham số quan trọng:

-   `--model`: Trỏ trực tiếp vào file GGUF.
    
-   `--tokenizer`: Trỏ vào thư mục `model-config` để vLLM nạp đúng quy tắc parse token và `chat_template` (ví dụ: ChatML).
    
-   `--max-model-len 4096`: Giới hạn context window để tiết kiệm VRAM, phù hợp với RAG (Context ~2000 tokens + Chat History ~1000 tokens).
    
-   `--gpu-memory-utilization 0.85`: Dành 85% VRAM cho model và KV Cache, chừa 15% cho các tác vụ OS.
    
-   `--served-model-name`: Đặt tên API, backend FastAPI sẽ gọi model này qua tên `pet-chat-model`.
    

## 4. API Tương thích OpenAI

Khi vLLM chạy lên, nó sẽ expose cổng 8000. Backend của bạn có thể gọi nó y hệt như gọi OpenAI API:

-   Endpoint: `http://vllm_engine:8000/v1/chat/completions`
    
-   Model: `pet-chat-model`
