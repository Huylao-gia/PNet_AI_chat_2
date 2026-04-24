
# CHI TIẾT GIAI ĐOẠN 2: KNOWLEDGE DISTILLATION & MULTI-TASK AUGMENTATION

**Mục tiêu:** Biến ~300 bài viết thô (Corpus) thành ~2300+ cặp hội thoại chất lượng cao, đa tác vụ để Fine-tune mô hình Llama-3.2-1B.

**Công cụ:** OpenAI API (`gpt-4o-mini`), Script Python bất đồng bộ (Asyncio) và kỹ thuật Data Augmentation.

## 2.1. Sinh dữ liệu qua API (API Generation & Personas)

Sử dụng Teacher Model (`gpt-4o-mini`) để đọc 300 bài viết thô và sinh ra các cặp Hỏi-Đáp. Để tránh bị lỗi Rate Limit (429 Too Many Requests), hệ thống sử dụng cơ chế **Exponential Backoff** và giới hạn luồng (Semaphore = 5).

**Chiến lược 4 Personas:**

-   **A. Direct QA (Hỏi đáp trực tiếp):** Câu hỏi rõ ràng, câu trả lời chính xác, khoa học.
    
-   **B. Panic & Urgent (Tình huống hoảng loạn):** Giả lập user viết tắt, lo lắng. Dạy model kỹ năng trấn an và khuyên đi thú y.
    
-   **C. Multi-turn (Đa lượt):** Dạy khả năng nhớ ngữ cảnh hội thoại.
    
-   **D. Negative QA (Từ chối an toàn):** Dạy model từ chối các yêu cầu kê đơn thuốc online để đảm bảo an toàn y khoa.
    

## 2.2. Kiểm định Chất lượng (LLM-as-a-Judge)

Các mẫu QA thô sinh ra từ bước 2.1 tiếp tục được gửi qua Teacher Model để chấm điểm độc lập.

-   **Tiêu chí (Thang 1-10):** Độ trung thực (Faithfulness) và Giọng điệu (Tone).
    
-   **Luật lọc:** Giữ lại các mẫu đạt **Điểm >= 7**.
    
-   **Xử lý Post-processing:** Tự động cắt bỏ các cụm từ máy móc như _"Dựa theo văn bản bạn cung cấp..."_
    
-   **Kết quả thực tế thu được:** **1181 mẫu QA** chất lượng cao (được chia cứng thành **1003 mẫu Train** và **178 mẫu Benchmark Test**).
    

## 2.3. Tăng cường Dữ liệu Đa tác vụ (Multi-task Data Augmentation)

Vì 1003 mẫu Train vẫn là mức khá rủi ro (dễ overfit) cho mô hình 1B, chiến lược **Tăng cường dữ liệu (Augmentation)** được áp dụng ngay tại local (không tốn API). Hệ thống tái sử dụng 1003 mẫu QA và 300 bài viết gốc để nhân bản thành 3 tác vụ học tập khác nhau:

-   **Tác vụ 1: RAG (Open-book QA) - ~1003 mẫu**
    
    -   _Input:_ `Thông tin tham khảo` + `Câu hỏi`
        
    -   _Output:_ Trả lời dựa sát vào thông tin.
        
    -   _Mục đích:_ Dạy model kỹ năng đọc hiểu văn bản (Context Reading) làm lõi cho hệ thống RAG sau này.
        
-   **Tác vụ 2: Closed-book QA (Hỏi đáp tự do) - ~1003 mẫu**
    
    -   _Input:_ Chỉ có `Câu hỏi` (Không có ngữ cảnh).
        
    -   _Output:_ Trả lời trực tiếp.
        
    -   _Mục đích:_ Dạy model ghi nhớ kiến thức nền tảng, có khả năng phản xạ tự nhiên ngay cả khi hệ thống Vector Search không tìm thấy tài liệu phù hợp.
        
-   **Tác vụ 3: Knowledge Internalization (Chia sẻ/Tóm tắt kiến thức) - ~300 mẫu**
    
    -   _Input:_ Tiêu đề bài viết ("Hãy chia sẻ kiến thức về...").
        
    -   _Output:_ Toàn bộ nội dung bài viết gốc (`content`).
        
    -   _Mục đích:_ Ép model học sâu cấu trúc ngữ pháp, từ vựng tiếng Việt chuyên ngành y tế và văn phong viết bài dài.
        

**-> Kết quả Tăng cường:** Nâng tổng số lượng tập Train lên **~2306 mẫu**. Đây là "điểm ngọt" (sweet spot) lý tưởng.

## 2.4. Formatting (Định dạng chuẩn Llama 3)

Mô hình `Llama-3.2-1B-Instruct` sử dụng template ChatML. Toàn bộ tập dữ liệu đa tác vụ được bọc trong các thẻ token chuẩn để mô hình không bị loạn.

**Cấu trúc ChatML mẫu:**

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Bạn là một chuyên gia thú y tận tâm...<|eot_id|><|start_header_id|>user<|end_header_id|>

Thông tin tham khảo: [Ngữ cảnh]
Câu hỏi: [Nội dung câu hỏi]<|eot_id|><|start_header_id|>assistant<|end_header_id|>

[Nội dung trả lời]<|eot_id|>

```

**Đầu ra cuối cùng (Outputs):**

-   `final_augmented_train.jsonl`: Chứa ~2306 mẫu Train đã format ChatML, sẵn sàng nạp vào QLoRA.
    
-   `final_benchmark.json`: Chứa 178 mẫu Test giữ nguyên định dạng thô ban đầu để dùng cho hệ thống đánh giá (Evaluation) tự động.
