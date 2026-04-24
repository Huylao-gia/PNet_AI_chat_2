import streamlit as st
import requests
import json
import uuid

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Pet Care Chatbot", page_icon="🐾", layout="centered")
st.title("🐾 Trợ lý Thú y Ảo (RAG Chatbot)")
st.caption("🤖 Hệ thống tư vấn sức khỏe thú cưng thông minh sử dụng tài liệu y khoa.")

# --- KHỞI TẠO SESSION STATE ---
# Tạo một session_id duy nhất cho mỗi phiên duyệt web để lưu lịch sử bên Backend
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Mảng lưu lịch sử hiển thị trên màn hình Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR CẤU HÌNH ---
with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    api_url = st.text_input("Backend API URL", value="http://localhost:8080/api/chat")
    top_k = st.slider("Số tài liệu tham chiếu (Top K)", min_value=1, max_value=5, value=3)
    
    st.divider()
    st.caption(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
    
    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        # Tạo session_id mới để Backend cũng xóa ngữ cảnh
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# --- HIỂN THỊ LỊCH SỬ CHAT ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- XỬ LÝ KHI NGƯỜI DÙNG NHẬP TIN NHẮN ---
if prompt := st.chat_input("Hỏi tôi về các vấn đề sức khỏe của chó, mèo..."):
    
    # 1. Hiển thị ngay câu hỏi của User lên màn hình
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Hiển thị khung chờ và Streaming câu trả lời từ AI
    with st.chat_message("assistant"):
        
        # Hàm generator để đọc luồng Server-Sent Events (SSE) từ Backend
        def stream_response():
            payload = {
                "session_id": st.session_state.session_id,
                "message": prompt,
                "top_k": top_k
            }
            try:
                # Gửi request với stream=True để đọc dữ liệu trả về liên tục
                response = requests.post(api_url, json=payload, stream=True)
                response.raise_for_status()
                
                # Duyệt qua từng dòng dữ liệu Backend đẩy về
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        # Kiểm tra chuẩn SSE format ("data: ...")
                        if decoded_line.startswith('data: '):
                            data_str = decoded_line[6:] # Cắt bỏ chữ 'data: ' ở đầu
                            
                            if data_str == '[DONE]':
                                break
                                
                            try:
                                # Parse JSON và lấy từng chữ
                                data_json = json.loads(data_str)
                                yield data_json.get("content", "")
                            except json.JSONDecodeError:
                                continue
            except requests.exceptions.ConnectionError:
                yield "❌ **Lỗi:** Không thể kết nối tới Backend. Hãy chắc chắn Server Backend đang chạy ở cổng 8080."
            except Exception as e:
                yield f"❌ **Lỗi xử lý:** {e}"

        # Lệnh st.write_stream tự động bắt generator và tạo hiệu ứng gõ chữ
        full_response = st.write_stream(stream_response)
        
    # 3. Lưu toàn bộ câu trả lời hoàn chỉnh vào bộ nhớ để hiển thị khi reload web
    st.session_state.messages.append({"role": "assistant", "content": full_response})
