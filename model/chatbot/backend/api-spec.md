
# 📚 TÀI LIỆU ĐẶC TẢ API - PET CARE RAG CHATBOT

Tài liệu này dành cho Web/Frontend Developer để tích hợp tính năng Trợ lý ảo (Chatbot) vào Website. API sử dụng giao thức **Server-Sent Events (SSE)** để truyền dữ liệu dạng luồng (streaming), giúp hiển thị từng chữ một (typing effect) giống như ChatGPT.

## 1. Thông Tin Chung (Overview)

-   **Base URL (Local):** `http://localhost:8080`
    
-   **Giao thức trả về:** `text/event-stream` (Server-Sent Events - SSE)
    
-   **CORS:** Đã được cấu hình mở (`*`), có thể gọi trực tiếp từ trình duyệt ở bất kỳ domain nào (Trong quá trình Dev).
    

## 2. Đặc Tả Endpoint: Gửi Tin Nhắn Chat

### `POST /api/chat`

Endpoint này nhận câu hỏi của người dùng và trả về câu trả lời của AI dưới dạng luồng dữ liệu liên tục (stream).

#### 2.1. Request Headers

**Header**

**Giá trị**

`Content-Type`

`application/json`

#### 2.2. Request Body (JSON Payload)

**Trường (Field)**

**Kiểu dữ liệu**

**Yêu cầu**

**Mô tả chi tiết**

`session_id`

`string`

**Bắt buộc**

Mã định danh duy nhất cho cuộc hội thoại. Dùng để Backend nhớ lịch sử chat. Frontend nên tự tạo ra một UUID và lưu vào `sessionStorage` hoặc `localStorage`.

`message`

`string`

**Bắt buộc**

Câu hỏi y khoa hoặc tin nhắn từ người dùng.

`top_k`

`integer`

_Tùy chọn_

Số lượng tài liệu tham khảo muốn Backend trích xuất. Mặc định là `3`. Thường Frontend không cần truyền trường này.

**Ví dụ Request Body:**

```
{
  "session_id": "user-session-12345-abcde",
  "message": "Chó nhà tôi bị ủ rũ và nôn mửa, tôi nên làm gì?"
}

```

#### 2.3. Cấu Trúc Trả Về (Response Stream)

Vì đây là API Streaming, response sẽ không trả về ngay lập tức một cục JSON lớn. Thay vào đó, kết nối sẽ được giữ mở, Server sẽ liên tục đẩy các dòng text (Chunks) về Frontend.

**Format chuẩn của một Chunk (Theo chuẩn SSE):**

Mỗi gói dữ liệu bắt đầu bằng chữ `data:` , tiếp theo là chuỗi JSON, và kết thúc bằng 2 dấu xuống dòng `\n\n`.

```
data: {"content": "Chào"}

data: {"content": " bạn,"}

data: {"content": " theo"}

data: {"content": " tài"}

data: {"content": " liệu..."}

```

**Tín hiệu kết thúc (End of Stream):**

Khi AI sinh xong câu trả lời, Server sẽ gửi một gói dữ liệu đặc biệt để báo hiệu Frontend đóng kết nối:

```
data: [DONE]

```

## 3. Hướng Dẫn Tích Hợp Frontend (Code Mẫu JavaScript)

Dưới đây là đoạn code JavaScript chuẩn mực sử dụng `fetch API` để xử lý SSE. Bạn có thể gắn nó vào nút "Gửi" (Send button) trên giao diện Chat Widget.

```
/**
 * Hàm gửi tin nhắn và đọc luồng phản hồi từ Chatbot API
 * @param {string} userMessage - Câu hỏi của người dùng
 * @param {function} onTokenReceived - Callback chạy mỗi khi nhận được 1 chữ mới
 * @param {function} onComplete - Callback chạy khi AI trả lời xong toàn bộ
 */
async function sendChatMessage(userMessage, onTokenReceived, onComplete) {
    // 1. Khởi tạo hoặc lấy Session ID từ bộ nhớ trình duyệt
    let sessionId = sessionStorage.getItem('chat_session_id');
    if (!sessionId) {
        sessionId = 'session_' + Math.random().toString(36).substring(2, 15);
        sessionStorage.setItem('chat_session_id', sessionId);
    }

    try {
        // 2. Gọi API bằng Fetch
        const response = await fetch('http://localhost:8080/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: sessionId,
                message: userMessage
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // 3. Đọc luồng dữ liệu (Stream Reader)
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');

        while (true) {
            const { value, done } = await reader.read();
            
            // Nếu stream đã bị đóng từ phía server
            if (done) break; 
            
            // Giải mã byte thành chuỗi text
            const chunkText = decoder.decode(value, { stream: true });
            
            // Server có thể gửi gộp nhiều dòng 'data: ...' trong 1 chunk
            // Cần tách ra bằng ký tự xuống dòng
            const lines = chunkText.split('\n');
            
            for (const line of lines) {
                // Chỉ lấy các dòng chứa dữ liệu SSE
                if (line.startsWith('data: ')) {
                    const dataStr = line.replace('data: ', '').trim();
                    
                    // Bỏ qua dòng trống
                    if (!dataStr) continue;
                    
                    // Nhận tín hiệu hoàn thành
                    if (dataStr === '[DONE]') {
                        if (onComplete) onComplete();
                        return;
                    }
                    
                    try {
                        // Phân tích JSON và lấy text
                        const jsonData = JSON.parse(dataStr);
                        if (jsonData.content) {
                            // Bắn chữ mới ra ngoài giao diện
                            onTokenReceived(jsonData.content);
                        }
                    } catch (e) {
                        console.error("Lỗi parse JSON chunk:", e, "Data:", dataStr);
                    }
                }
            }
        }
    } catch (error) {
        console.error("Lỗi gọi API Chat:", error);
        onTokenReceived("\n\n*Xin lỗi, hệ thống đang bận hoặc mất kết nối.*");
        if (onComplete) onComplete();
    }
}

// ==========================================
// CÁCH SỬ DỤNG TRÊN GIAO DIỆN (Ví dụ)
// ==========================================
/*
const chatBox = document.getElementById('chat-content');
let currentAiMessageDiv = null;

// Khi người dùng bấm nút GỬI
function handleSend() {
    const text = "Chó nhà tôi bị nôn mửa, phải làm sao?";
    
    // In câu hỏi lên màn hình...
    
    // Tạo 1 thẻ div trống để hứng câu trả lời của AI
    currentAiMessageDiv = document.createElement('div');
    chatBox.appendChild(currentAiMessageDiv);
    
    // Gọi API
    sendChatMessage(
        text, 
        // Callback 1: Nối dần chữ vào div (Tạo hiệu ứng gõ phím)
        (token) => {
            currentAiMessageDiv.innerHTML += token;
            // Tự động cuộn xuống dưới cùng
            chatBox.scrollTop = chatBox.scrollHeight;
        },
        // Callback 2: Khi hoàn thành
        () => {
            console.log("AI đã trả lời xong!");
            // Tại đây có thể render lại nội dung bằng thư viện Markdown (như marked.js)
        }
    );
}
*/

```

## 4. Best Practices (Lưu ý cho Frontend)

1.  **Quản lý `session_id`:** Hệ thống dùng `session_id` để "nhớ" ngữ cảnh. Nếu Website muốn người dùng reset lại cuộc trò chuyện (Nút "Chat mới"), Frontend chỉ cần tạo ra một chuỗi `session_id` mới ngẫu nhiên và gửi lên.
    
2.  **Render Markdown:** AI (vLLM/OpenAI) trả về text có định dạng Markdown (ví dụ: `**Chữ in đậm**`, `- Gạch đầu dòng`). Giao diện Chat của bạn nên tích hợp thư viện như `marked.js` hoặc `react-markdown` để chuyển đổi chuỗi text stream thành HTML hiển thị đẹp mắt.
    
3.  **Scroll to bottom:** Khi streaming, chữ sẽ dài ra liên tục, hãy nhớ cập nhật hàm cuộn màn hình (`scrollTop`) mỗi khi nhận được token mới.
    
4.  **Timeouts:** Vì RAG cần thời gian lục lọi cơ sở dữ liệu trước khi sinh chữ đầu tiên, khoảng thời gian chờ (Time-to-first-token) có thể mất 1-3 giây. Frontend nên hiển thị "AI đang suy nghĩ..." (Typing indicator) trước khi nhận được token đầu tiên để UX mượt mà hơn.
