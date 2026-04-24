
# KẾ HOẠCH CHI TIẾT: DATA PIPELINE & VECTOR DATABASE

## 1. Lựa chọn Vector Database

-   **Yêu cầu:** Nhỏ, nhẹ, nhanh, tối ưu cho dự án nội bộ/quy mô vừa.
    
-   **Quyết định:** **ChromaDB (Embedded Mode)**.
    
-   **Lý do:** ChromaDB chạy trực tiếp dưới dạng thư viện Python (nhúng vào FastAPI) và lưu dữ liệu ra file local (`.sqlite3` và `.parquet`). Không cần chạy thêm 1 container riêng biệt (như Milvus hay Qdrant), giúp tiết kiệm RAM và CPU.
    

## 2. Lựa chọn Embedding Model

-   **Model ưu tiên:** `keepitreal/vietnamese-sbert` (Đã đề cập trong outline).
    
-   **Đặc điểm:** Tối ưu hóa cho ngữ nghĩa tiếng Việt, kích thước model nhỏ (~400MB), có thể chạy rất nhanh trên CPU mà không tranh giành GPU với vLLM.
    

## 3. Kiến trúc Data Processing Pipeline (Từ PDF đến ChromaDB)

Cần xây dựng một script độc lập `ingest_data.py` để chạy offline (chỉ chạy khi có dữ liệu mới).

### Bước 3.1: Extract Text từ PDF

-   **Công cụ:** Dùng thư viện `PyMuPDF` (`fitz`). Tốc độ đọc PDF nhanh nhất hiện nay, giữ nguyên được cấu trúc đoạn văn.
    
-   **Xử lý:** Xóa bỏ header/footer, bảng biểu nhiễu, các ký tự thừa.
    

### Bước 3.2: Chunking (Cắt đoạn)

-   **Công cụ:** `RecursiveCharacterTextSplitter` của LangChain.
    
-   **Cấu hình:** - `chunk_size = 500` tokens (hoặc từ).
    
    -   `chunk_overlap = 50` tokens (để giữ ngữ cảnh liền mạch giữa 2 đoạn bị cắt).
        

### Bước 3.3: Metadata Enrichment

Mỗi chunk khi đưa vào ChromaDB không chỉ có nội dung mà cần có Metadata:

```
{
  "source": "ten_file_pdf.pdf",
  "page": 12,
  "topic": "benh_duong_ruot_o_cho" 
}

```

### Bước 3.4: Khởi tạo & Lưu trữ vào ChromaDB

**Mô hình thư mục sinh ra:** `/chatbot/data/chroma_db/`

**Mã giả cốt lõi (Pseudo-code cho thư viện nhúng):**

```
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# Khởi tạo embedding
embedder = SentenceTransformer('keepitreal/vietnamese-sbert')

# Khởi tạo DB
client = PersistentClient(path="/chatbot/data/chroma_db")
collection = client.get_or_create_collection(
    name="pet_medical_docs",
    metadata={"hnsw:space": "cosine"} # Dùng độ đo Cosine Similarity
)

# Chèn dữ liệu
collection.add(
    documents=["Đoạn text 1", "Đoạn text 2"],
    metadatas=[{"source": "doc1.pdf"}, {"source": "doc2.pdf"}],
    ids=["chunk_1", "chunk_2"],
    embeddings=embedder.encode(["Đoạn text 1", "Đoạn text 2"]).tolist()
)

```
