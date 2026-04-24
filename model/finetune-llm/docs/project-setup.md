
# HƯỚNG DẪN THIẾT LẬP MÔI TRƯỜNG & CẤU TRÚC DỰ ÁN

**Project:** Pet Care RAG Chatbot & LLM Fine-tuning

**Version:** 1.0.0

Tài liệu hướng dẫn cách tổ chức thư mục dự án chuẩn, cách thiết lập môi trường Python ảo và cài đặt các dependencies cần thiết cho toàn bộ quá trình từ xử lý dữ liệu đến huấn luyện mô hình.

## 1. PROJECT STRUCTURE

Khởi tạo thư mục trong thư mục gốc (`/`) của dự án theo đúng cấu trúc dưới đây. Việc tách rõ ràng source code (`src/`) và dữ liệu (`data/`) giúp tránh xung đột và dễ dàng push Git (bỏ qua thư mục data).

```
pet-rag-chatbot/
├── data/                       # Chứa toàn bộ dữ liệu (KHÔNG push lên Git)
│   ├── raw/                    # Dữ liệu gốc chưa xử lý (2viet.json, papddy.json)
│   ├── processed/              # Dữ liệu sau Giai đoạn 1 (cleaned_corpus.json)
│   └── distillation/           # Dữ liệu Hỏi-Đáp QA (train_ready.jsonl, benchmark.json)
├── docs/                       # Tài liệu dự án (pipeline, blueprints, setup)
│   ├── finetune-pipeline.md    # Bản vẽ kỹ thuật & Kế hoạch tổng thể
│   └── architecture.png        # (Tùy chọn) Sơ đồ kiến trúc RAG
├── src/                        # Source code chính của dự án
│   ├── __init__.py
│   ├── data_pipeline/          # Code Giai đoạn 1 (Làm sạch, merge)
│   │   └── clean_data.py       # (Script data_pipeline.py đã viết)
│   ├── distillation/           # Code Giai đoạn 2 (Sinh dữ liệu QA)
│   │   └── qa_generator.py     # Script gọi API Teacher Model
│   ├── finetune/               # Code Giai đoạn 3 (Huấn luyện LLM)
│   │   └── unsloth_train.py    # Script cấu hình QLoRA
│   └── serving/                # Code Giai đoạn 4 (Triển khai vLLM API)
│       └── vllm_runner.sh      # Script khởi chạy Docker vLLM
├── notebooks/                  # Các file Jupyter Notebook để test nhanh hoặc EDA
│   └── 01_eda_data.ipynb
├── .env.example                # File mẫu chứa các biến môi trường (API Keys)
├── .gitignore                  # Cấu hình bỏ qua file không cần push lên Github
└── requirements.txt            # Danh sách thư viện Python cần cài đặt

```

## 2. VIRTUAL ENVIRONMENT

Dự án AI yêu cầu quản lý phiên bản thư viện rất chặt chẽ, đặc biệt là `torch` (PyTorch) và thư viện CUDA để tương thích với GPU.

**Yêu cầu hệ thống:**

-   Hệ điều hành: Linux (Ubuntu 22.04 khuyến nghị) hoặc Windows (dùng WSL2).
    
-   Python: Phiên bản **3.10** hoặc **3.11** (Tránh dùng 3.12 vì một số thư viện C++ biên dịch AI chưa hỗ trợ hoàn toàn).
    

**Các bước cài đặt:**

1.  **Tạo môi trường ảo (Virtual Environment):**
    
    Mở terminal tại thư mục gốc của dự án và chạy:
    
    ```
    python3 -m venv venv
    
    ```
    
2.  **Kích hoạt môi trường ảo:**
    
    -   Trên Linux/Mac/WSL:
        
        ```
        source venv/bin/activate
        
        ```
        
    -   Trên Windows (Command Prompt/PowerShell):
        
        ```
        venv\Scripts\activate
        
        ```
        
3.  **Nâng cấp `pip`:**
    
    ```
    pip install --upgrade pip
    
    ```
    

## 3. CÀI ĐẶT THƯ VIỆN

Tạo file `requirements.txt` ở thư mục gốc với nội dung cơ bản sau:

```
# --- Data Engineering & EDA ---
pandas==2.2.0
matplotlib==3.8.2
seaborn==0.13.2
regex==2023.12.25

# --- Knowledge Distillation ---
openai==1.12.0          # Dùng gọi API GPT-4 hoặc model tương thích OpenAI
anthropic==0.19.1       # (Tùy chọn) Nếu dùng Claude làm Teacher Model
tqdm==4.66.2            # Thanh tiến trình (Progress bar)

# --- Thư viện hệ thống & Tiện ích ---
python-dotenv==1.0.1    # Quản lý biến môi trường .env
pydantic==2.6.1         # Validation cấu trúc dữ liệu JSON

```

_Lưu ý:_ Đối với các thư viện huấn luyện nặng như `torch`, `xformers` hay `unsloth`, KHÔNG đưa vào requirements gốc ngay từ đầu vì nó phụ thuộc vào loại GPU máy tính bạn đang dùng. Sẽ cài đặt chúng trong môi trường GPU cụ thể ở Giai đoạn 3.

**Cài đặt các gói trên bằng lệnh:**

```
pip install -r requirements.txt

```

## 4. ENVIRONMENT VARIABLES

Trong Giai đoạn 2 và 3 sẽ cần gọi API (OpenAI/Anthropic) và kết nối với HuggingFace. Để bảo mật, KHÔNG bao giờ hard-code API Key vào mã nguồn.

1.  Tạo một file có tên `.env` ở thư mục gốc dự án.
    
2.  Mở file `.env` và thêm các khóa sau (thay thế bằng key thật của bạn):
    

```
# Teacher Model API Keys (Cho Giai đoạn Distillation)
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# HuggingFace Token (Dùng tải model Llama 3 và upload model sau khi Fine-tune)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Cấu hình đường dẫn nội bộ
RAW_DATA_PATH=data/raw/
PROCESSED_DATA_PATH=data/processed/
DISTILLATION_DATA_PATH=data/distillation/

```

3.  Trong file `.gitignore`, hãy chắc chắn bạn đã thêm `.env` để tránh vô tình đẩy mã bí mật lên mạng:
    

```
# .gitignore
venv/
__pycache__/
.env
data/
*.ipynb_checkpoints

```

**Xác nhận thành công:** Khi bạn đã setup xong cấu trúc này, bạn có thể di chuyển file `data_pipeline.py` tôi đã viết cho bạn vào thư mục `src/data_pipeline/`, đặt `2viet.json` và `papddy.json` vào `data/raw/` và chạy thử lệnh:

`python src/data_pipeline/data_pipeline.py`
