
# BLUEPRINT TỔNG THỂ: KIẾN TRÚC & TRIỂN KHAI RAG CHATBOT (GIAI ĐOẠN 4)

**Dự án:** Pet Care RAG Chatbot

**Phiên bản:** 1.0.0 (Production-Ready)

**Mục tiêu:** Đóng gói mô hình ngôn ngữ (LLM) và Cơ sở dữ liệu Vector thành các Microservices độc lập chạy trên Docker. Cung cấp API Streaming chuẩn OpenAI cho Front-end (Website) tiêu thụ.

## 1. TỔNG QUAN KIẾN TRÚC HỆ THỐNG (SYSTEM ARCHITECTURE)

Hệ thống được thiết kế theo mô hình **Decoupled Microservices** (Phân tách dịch vụ), bao gồm 2 Container hoạt động song song bên trong một mạng nội bộ (Docker Bridge Network).

### 1.1. Luồng Dữ Liệu (Data Flow)

1.  **User (Website):** Người dùng gõ câu hỏi: _"Chó nhà em nôn mửa liên tục"_. Front-end gọi `POST /api/chat` tới cổng 8080.
    
2.  **RAG Backend (Cổng 8080):** Nhận request, tạo embedding cho câu hỏi và truy vấn vào ChromaDB. Lấy ra top 2 bài viết y khoa phù hợp nhất.
    
3.  **RAG Backend -> LLM Engine:** Backend ghép tài liệu (Context) và câu hỏi thành 1 Prompt chuẩn ChatML, sau đó gọi HTTP POST sang vLLM Engine (Cổng 8000 mạng nội bộ).
    
4.  **LLM Engine (vLLM):** Xử lý Prompt trên GPU, sinh ra từng Token (chữ) và nhả về cho Backend.
    
5.  **RAG Backend -> User:** Sử dụng giao thức SSE (Server-Sent Events), Backend đẩy ngay lập tức từng Token về cho Website để hiển thị hiệu ứng gõ chữ (Streaming).
    

## 2. THIẾT KẾ CÁC DỊCH VỤ (SERVICE DESIGN)

### 🐳 Service 1: Core LLM Engine (Trái tim hệ thống)

-   **Nền tảng:** `vLLM` (Sử dụng Image Docker chính chủ `vllm/vllm-openai`).
    
-   **Vai trò:** Cung cấp năng lực tính toán thuần túy. Biến file trọng số GGUF thành một API Server tương thích 100% với chuẩn của OpenAI.
    
-   **Đặc tính Kỹ thuật:**
    
    -   Khả năng xử lý hàng loạt liên tục (Continuous Batching).
        
    -   Tối ưu bộ nhớ với PagedAttention (tránh tràn VRAM khi có nhiều user).
        
    -   **Chỉ mở cổng trong mạng nội bộ**, tuyệt đối không public ra Internet để bảo mật.
        

### 🐳 Service 2: RAG Backend & API Gateway (Bộ não điều phối)

-   **Nền tảng:** `Python 3.10` + `FastAPI`.
    
-   **Vai trò:** Xử lý nghiệp vụ (Business Logic), quản lý VectorDB, và bảo vệ LLM Engine.
    
-   **Các thành phần lõi:**
    
    -   **Vector Database:** `ChromaDB` (chạy dạng thư viện nhúng - persistent storage).
        
    -   **Embedding Model:** `keepitreal/vietnamese-sbert` (Chạy trên CPU, dùng để mã hóa tiếng Việt).
        
    -   **Giao thức API:** `RESTful` kết hợp `SSE (Server-Sent Events)`.
        
-   **Tại sao lại dùng REST + SSE mà không dùng WebSockets?**
    
    -   SSE là giao thức 1 chiều từ Server -> Client, hoàn hảo cho việc stream text từ LLM.
        
    -   Stateless (Không lưu trạng thái), dễ dàng cấu hình qua NGINX và dễ cân bằng tải (Load Balancing) hơn WebSocket.
        

## 3. CẤU TRÚC THƯ MỤC TRIỂN KHAI (DIRECTORY STRUCTURE)

Dự án trên máy chủ (VPS) sẽ được tổ chức thành một khối thống nhất:

```
pet_rag_server/
├── data/                               # Dữ liệu mount vào Docker (Persistent Volume)
│   ├── models/
│   │   └── pet_chatbot_q4.gguf         # File GGUF (Đã train ở GĐ3)
│   └── vector_db/                      # File DB sinh ra từ ChromaDB
├── backend/                            # Thư mục chứa code Service 2
│   ├── Dockerfile                      # Kịch bản build image cho FastAPI
│   ├── requirements.txt                # Thư viện (fastapi, chromadb, openai,...)
│   └── main.py                         # Mã nguồn RAG logic
└── docker-compose.yml                  # Kịch bản Orchestration khởi chạy toàn hệ thống

```

## 4. YÊU CẦU PHẦN CỨNG & PHẦN MỀM (INFRASTRUCTURE PRE-REQUISITES)

-   **Hệ điều hành:** Ubuntu 22.04 LTS.
    
-   **Phần mềm:**
    
    -   Docker Engine & Docker Compose (v2+).
        
    -   `nvidia-container-toolkit` (Bắt buộc để Docker nhận diện được GPU).
        
-   **Phần cứng:**
    
    -   **GPU:** Tối thiểu 1 GPU NVIDIA (vd: T4 16GB, RTX 3060 12GB trở lên).
        
    -   **RAM:** Tối thiểu 16GB.
        
    -   **Ổ cứng:** SSD > 50GB (Để chứa các image Docker và Vector DB).
        

## 5. HƯỚNG DẪN TRIỂN KHAI TỪNG BƯỚC (DEPLOYMENT PLAYBOOK)

### Bước 1: Chuẩn bị Artifacts

1.  Tải file model `.gguf` từ Google Drive và đặt vào `pet_rag_server/data/models/`.
    
2.  Tải file `cleaned_corpus.json` về một thư mục tạm, chạy script tạo ChromaDB (đã làm) để sinh ra thư mục DB, sau đó copy toàn bộ vào `pet_rag_server/data/vector_db/`.
    

### Bước 2: Khởi chạy Hệ thống

Truy cập vào thư mục `pet_rag_server/` và thực thi lệnh:

```
docker-compose up -d --build

```

### Bước 3: Kiểm tra Sức khỏe (Health Check)

Xem log của hệ thống để đảm bảo vLLM đã nạp thành công model vào VRAM và FastAPI đã kết nối được ChromaDB:

```
docker-compose logs -f

```

### Bước 4: Kiểm thử API (cURL)

Mở một terminal mới và giả lập request từ Website:

```
curl -N -X POST http://localhost:8080/api/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Dấu hiệu nhận biết chó bị dại là gì?"}'

```

_(Cờ `-N` giúp cURL không buffer dữ liệu mà sẽ in ra màn hình từng chữ ngay lập tức - xác nhận SSE đang hoạt động)._

## 6. KẾ HOẠCH BẢO TRÌ VÀ NÂNG CẤP (FUTURE SCALING)

-   **Cập nhật kiến thức:** Khi có kiến thức thú y mới, chỉ cần chạy lại script tạo Embedding chèn thêm vào thư mục `vector_db` và khởi động lại Backend Container. Không cần train lại LLM.
    
-   **Bảo mật API:** Bổ sung Middleware kiểm tra `API_KEY` hoặc `JWT Token` vào file `main.py` của FastAPI trước khi mở public port 8080 ra ngoài internet, tránh bị gọi lén API.
    
-   **Cân bằng tải:** Nếu lượng người dùng tăng đột biến, thiết lập NGINX làm Reverse Proxy đứng trước cổng 8080 để quản lý Rate Limit và phân luồng traffic.
