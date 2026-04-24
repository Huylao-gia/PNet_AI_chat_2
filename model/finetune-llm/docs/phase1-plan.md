
# CHI TIẾT GIAI ĐOẠN 1: DATA ENGINEERING (XỬ LÝ DỮ LIỆU THÔ)

**Dự án:** Pet Care RAG Chatbot **Nguồn dữ liệu gốc:** Crawl/Scrape từ các website về thú cưng (`2viet.json` và `papddy.json`). **Mục tiêu cốt lõi:** Chuyển đổi dữ liệu thô, nhiều rác HTML/quảng cáo thành một bộ Corpus (tập văn bản) tiếng Việt sạch, chuẩn hóa, sẵn sàng cho việc trích xuất tri thức (Knowledge Distillation) và xây dựng cơ sở dữ liệu Vector (Vector DB) cho RAG.

## 1.1. Hợp nhất và Loại bỏ Trùng lặp (Data Consolidation)

Dữ liệu thu thập từ nhiều nguồn thường có sự chồng chéo (cùng một bài viết được đăng trên nhiều trang khác nhau). Việc để mô hình học hoặc truy xuất dữ liệu trùng lặp sẽ gây lãng phí tài nguyên tính toán.

-   **Dữ liệu Đầu vào (Inputs):** * `data/raw/2viet.json` (List of JSON objects: `url, title, content, tag`).
    
    -   `data/raw/papddy.json` (List of JSON objects).
        
-   **Quy trình Thực thi:**
    
    1.  **Gộp dữ liệu:** Đọc và merge 2 list JSON lại với nhau. Thêm một trường metadata `source` ("2viet" hoặc "papddy") vào mỗi object để đảm bảo khả năng truy vết (Traceability).
        
    2.  **Tạo mã băm (Hashing):** Sử dụng thuật toán `MD5` để tạo mã băm cho chuỗi `Title + Content`. (Lý do: Không dùng URL để check trùng vì một bài viết có thể nằm ở 2 URL khác nhau).
        
    3.  **Lọc trùng lặp (Deduplication):** So sánh các mã băm, chỉ giữ lại bản ghi xuất hiện lần đầu tiên. Bỏ qua các bản ghi có nội dung rỗng (Null/Empty).
        
-   **Đầu ra (Outputs):** `data/processed/merged_raw.json`
    

## 1.2. Tiền xử lý & Làm sạch Dữ liệu (Data Cleaning & Preprocessing)

Đây là bước quan trọng nhất để tránh tình trạng "Garbage In, Garbage Out". Mô hình LLM RAG sẽ trả lời y hệt những gì nó đọc được, do đó tuyệt đối không để lọt quảng cáo vào tập dữ liệu.

-   **Dữ liệu Đầu vào:** `merged_raw.json`
    
-   **Quy trình Thực thi (Các lớp lọc Regex):**
    
    1.  **Chuẩn hóa Unicode:** Đưa toàn bộ văn bản tiếng Việt về chuẩn Unicode dựng sẵn (NFC) để Tokenizer của LLM không bị cắt token sai.
        
    2.  **Loại bỏ Markup:** Xóa sạch các thẻ HTML (`<p>`, `<div>`, `<a>`,...) và cú pháp Markdown rác (như cú pháp chèn ảnh `![image](url)`).
        
    3.  **Lọc Quảng cáo (Call-to-Action):** Sử dụng Biểu thức chính quy (Regex) để xóa bỏ các câu kêu gọi mua hàng thường nằm ở cuối bài.
        
        -   _Ví dụ các cụm bị xóa:_ "Liên hệ ngay hotline 09...", "Mua hàng tại link...", "Truy cập website để biết thêm...".
            
    4.  **Chuẩn hóa Teencode & Viết tắt:** Sửa các lỗi viết tắt phổ biến trong ngành (Ví dụ: "bsi", "bs" -> "bác sĩ"; "ko" -> "không"; "sp" -> "sản phẩm").
        
    5.  **Làm sạch khoảng trắng:** Loại bỏ các dấu cách thừa, các dấu chấm/phẩy lặp lại nhiều lần.
        
    6.  **Lọc theo độ dài:** Chỉ giữ lại các bài viết có nội dung sau khi làm sạch dài hơn 50 ký tự (loại bỏ các bài viết rác, quá ngắn không có giá trị thông tin).
        
-   **Đầu ra (Outputs):** `data/processed/cleaned_corpus.json` (Đây là file "vàng" sẽ được dùng xuyên suốt cho các giai đoạn sau).
    

## 1.3. Khám phá Dữ liệu (EDA - Exploratory Data Analysis)

Sau khi có dữ liệu sạch, cần phân tích thống kê để đưa ra các quyết định cấu hình cho mô hình huấn luyện và hệ thống RAG.

-   **Dữ liệu Đầu vào:** `cleaned_corpus.json`
    
-   **Quy trình Phân tích:**
    
    1.  **Phân tích Phân bố Tag (Tag Distribution):**
        
        -   _Hành động:_ Vẽ biểu đồ Barplot thống kê top 20 Tags phổ biến nhất.
            
        -   _Mục đích:_ Phát hiện sự mất cân bằng dữ liệu (Ví dụ: Bài viết về Chó quá nhiều, bài viết về Mèo quá ít). Từ đó lên kế hoạch dùng AI để tạo thêm dữ liệu (Augmentation) bù đắp cho các mảng kiến thức bị thiếu.
            
    2.  **Phân tích Độ dài Token (Token Length Distribution):**
        
        -   _Hành động:_ Ước tính số lượng Token của mỗi bài viết bằng công thức `Số từ (words) / 0.75` (Do tiếng Việt tốn nhiều token hơn). Vẽ biểu đồ Histogram phân bố.
            
        -   _Mục đích:_ Tìm ra chiều dài trung bình và tối đa của các bài viết. Thông số này cực kỳ quan trọng để:
            
            -   Cấu hình `max_seq_length` khi Fine-tune (đảm bảo model không bị cắt cụt câu khi học).
                
            -   Quyết định `chunk_size` (kích thước cắt đoạn văn bản) khi đưa dữ liệu vào Vector Database (Qdrant/ChromaDB) ở Giai đoạn 4.
                
-   **Đầu ra (Outputs):**
    
    -   Biểu đồ: `data/processed/eda_tag_distribution.png`
        
    -   Biểu đồ: `data/processed/eda_token_length.png`
        
    -   Báo cáo log tóm tắt (Tổng số bài, số token min/max/mean).
        

**Công cụ & Thư viện sử dụng:** `Python 3`, `pandas` (xử lý dataframe), `matplotlib` & `seaborn` (vẽ biểu đồ), `re` (Regex), `hashlib`.
