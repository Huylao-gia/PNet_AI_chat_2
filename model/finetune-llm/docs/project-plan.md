# BLUEPRINT: KẾ HOẠCH XÂY DỰNG PIPELINE DỮ LIỆU & FINE-TUNE LLM CHO CHATBOT THÚ CƯNG

**Dự án:** Pet Care RAG Chatbot

**Mô hình target:** `phamhai/Llama-3.2-1B-Instruct-Frog`

**Vai trò tài liệu:** Kiến trúc tổng thể (Blueprint), chiến lược Distillation quy mô lớn & Ước lượng thời gian triển khai.

## GIAI ĐOẠN 1: DATA ENGINEERING (XỬ LÝ DỮ LIỆU THÔ)

**⏱️ ETA (Thời gian dự kiến): 1 - 2 ngày** _(Với sự hỗ trợ của AI viết script Python)_

### 1.1. Hợp nhất Dữ liệu (Data Consolidation)

-   **Hiện trạng:** 2 files `2viet.json`, `papddy.json` chứa các object `{url, title, content, tag}`.
    
-   **Hành động:** Merge 2 file, thêm trường `source` ("2viet" hoặc "papddy"), loại bỏ trùng lặp (Deduplication) qua mã băm (hashing).
    

### 1.2. Tiền xử lý & Làm sạch Dữ liệu (Data Cleaning & Preprocessing)

-   **Hành động:**
    
    -   Xóa mã markup (HTML/Markdown rác).
        
    -   Chuẩn hóa teencode, viết tắt ("bsi" -> "bác sĩ"), chuẩn hóa bảng mã Unicode.
        
    -   Xóa nội dung quảng cáo (Call-to-action).
        
-   **Đầu ra:** `cleaned_corpus.json`.
    

### 1.3. Khám phá Dữ liệu (EDA)

-   **Hành động:** Phân tích phân bố Tag (Chó, Mèo, Bệnh, Dinh dưỡng...) để tìm lỗ hổng dữ liệu. Tính độ dài Token trung bình để chọn tham số context length khi huấn luyện.
    

## GIAI ĐOẠN 2: LARGE-SCALE KNOWLEDGE DISTILLATION & DATA CURATION

**⏱️ ETA (Thời gian dự kiến): 2 - 4 ngày** _(Tùy thuộc vào tốc độ gọi API của Teacher Model)_

_Mục tiêu: Không chỉ trích xuất Hỏi-Đáp đơn giản, mà phải tổng hợp (synthesize) một lượng dữ liệu đủ lớn (khoảng 5,000 - 10,000 samples) bao phủ mọi ngóc ngách hội thoại, đồng thời có cơ chế tự động lọc rác._

### 2.1. Chiến lược Chắt lọc & Mở rộng Dữ liệu (Data Augmentation & Distillation)

Thay vì chỉ dùng 1 prompt đơn điệu, chúng ta thiết kế hệ thống **Multi-Persona Prompting** dùng Teacher Model (GPT-4o-mini hoặc Claude 3.5 Haiku):

-   **Kịch bản 1: QA Trực tiếp (Direct Fact QA):** Tạo câu hỏi trực diện từ `content`. (VD: "Triệu chứng bệnh Parvo là gì?").
    
-   **Kịch bản 2: Tình huống Khẩn cấp (Panic Persona):** Mô phỏng người dùng đang hoảng loạn, gõ sai chính tả, không chấm phẩy. Model học cách giữ bình tĩnh, xoa dịu và đưa ra lời khuyên đi bác sĩ (Rất quan trọng cho bot y tế).
    
-   **Kịch bản 3: Hỏi đáp Đa lượt (Multi-turn Conversation):** Xây dựng hội thoại 2-3 lượt (VD: Lượt 1 hỏi triệu chứng -> Bot trả lời -> Lượt 2 hỏi về chi phí/cách chăm sóc tại nhà).
    
-   **Kịch bản 4: Negative QA (Từ chối trả lời an toàn):** Tạo các câu hỏi vượt ngoài phạm vi thú y (VD: Khám bệnh cho người) hoặc yêu cầu kê đơn thuốc đặc trị (Bot RAG không được phép kê đơn, phải khuyên đi viện).
    

### 2.2. Kiểm định Chất lượng Tự động (LLM-as-a-Filter)

Dữ liệu sinh ra từ AI sẽ có "ảo giác". Cần một bước chạy qua một LLM khác (hoặc prompt hệ thống đánh giá ngặt nghèo) để chấm điểm từng cặp QA:

-   **Tiêu chí:**
    
    1.  _Độ trung thực (Faithfulness):_ Câu trả lời có đi chệch khỏi `cleaned_corpus` gốc không?
        
    2.  _Tone giọng (Tone):_ Đã đủ thân thiện, đồng cảm và chuyên nghiệp chưa?
        
    3.  _An toàn (Safety):_ Có lỡ kê đơn thuốc tùy tiện không?
        
-   **Hành động:** Loại bỏ tự động các cặp QA có điểm số < 8/10.
    

### 2.3. Định dạng Dữ liệu (Formatting)

-   Chuyển đổi tập QA hoàn chỉnh (khoảng 85% Train, 15% Benchmark) sang định dạng ChatML/Llama-3 template (có `<|start_header_id|>user...`).
    
-   **Đầu ra cuối:** `train_large_ready.jsonl` và `benchmark_dataset.json`.
    

## GIAI ĐOẠN 3: CHIẾN LƯỢC HUẤN LUYỆN (FINE-TUNING)

**⏱️ ETA (Thời gian dự kiến): 1 - 2 ngày** _(Bao gồm setup, test run và full run)_

### 3.1. Phương pháp & Cấu hình (QLoRA)

-   Sử dụng **QLoRA** kết hợp thư viện **Unsloth** để tối ưu hóa VRAM và x2 tốc độ huấn luyện.
    
-   **Tối ưu hóa Hyperparameters cho dữ liệu lớn:**
    
    -   **Rank (r):** Nâng lên 32 hoặc 64 (Vì dữ liệu đã phức tạp và đa dạng hơn, cần Rank cao hơn để học được nhiều nơ-ron ngữ cảnh).
        
    -   **Alpha:** 64 hoặc 128 (Thường gấp đôi Rank).
        
    -   **Learning Rate:** 2e-4 hoặc sử dụng Cosine Scheduler có warmup.
        
    -   **Epochs:** 1 đến 3. (Cài đặt Early Stopping dựa trên Evaluation Loss trên tập Benchmark để tự ngắt khi model bắt đầu học vẹt/overfit).
        

### 3.2. Đánh giá Mô hình (Evaluation)

-   Chạy lại tập `benchmark_dataset.json` trên model gốc và model sau Fine-tune.
    
-   Sử dụng các metrics của hệ thống RAGAS (Answer Relevance, Context Precision) và Teacher Model mù để so sánh.
    

## GIAI ĐOẠN 4: DEPLOYMENT & API SERVING (HOSTING)

**⏱️ ETA (Thời gian dự kiến): 1 - 2 ngày** _(Tùy độ quen thuộc với Docker và Linux)_

### 4.1. Lựa chọn Hosting (Đề xuất: vLLM)

-   **Phương tiện:** Sử dụng **vLLM** qua Docker trên máy ảo GPU (NVIDIA T4 16GB hoặc RTX 3090/4090).
    
-   **Ưu điểm tuyệt đối:** Hỗ trợ _PagedAttention_ giúp giảm lãng phí VRAM từ 50% xuống còn 4%, hỗ trợ Continuous Batching xử lý hàng chục user cùng lúc mà không sụt giảm TPS (Tokens Per Second).
    

### 4.2. Chế độ Host & Tích hợp

-   **Mode (Chế độ):** Chạy Server ở dạng **OpenAI-Compatible API** + Kích hoạt tính năng **Streaming (Server-Sent Events - SSE)**. Điều này giúp tích hợp liền mạch vào UI website, chữ hiện ra từng từ giống hệt ChatGPT (giảm cảm giác chờ đợi).
    
-   **System Prompt Core:** Thiết lập sẵn System Prompt mặc định trên API Server: _"Bạn là một chuyên gia thú y tận tâm. Hãy trả lời dựa trên ngữ cảnh được cung cấp..."_
    

**TỔNG THỜI GIAN ƯỚC TÍNH (END-TO-END PIPELINE): Khoảng 5 đến 10 ngày làm việc.** _Sử dụng AI Core (như Claude 3.5 Sonnet hoặc GPT-4o) để viết toàn bộ script xử lý dữ liệu và prompt tạo data sẽ giúp bạn tiết kiệm ít nhất 2 tuần code tay thông thường._
