import fitz  # PyMuPDF
import re
import os
import json
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# --- CẤU HÌNH ---
# Thư mục chứa toàn bộ dữ liệu bạn đã crawl về (Cả PDF và JSON)
CRAWL_DIR = r"C:\Users\USER\Documents\FPT_9\Crawl\File_pdf_json"
DB_PATH = "chroma_db"
COLLECTION_NAME = "pet_medical_docs"
EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"

# --- 1. GIỮ NGUYÊN HÀM CLEAN SÂU CỦA BẠN (Rất quan trọng cho PDF tiếng Việt) ---
def clean_medical_text(text):
    if not text: return ""
    # Xóa ký tự điều khiển
    text = re.sub(r'[\x00-\x09\x0b-\x1f\x7f-\x9f]', '', text)
    # Xóa số trang
    text = re.sub(r'^\s*-\s*\d+\s*-\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # Sửa lỗi ngắt dòng giữa câu (giữ nguyên logic của bạn)
    vietnamese_lower = "a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    text = re.sub(rf'(?<=[^\.\!\?\:\;])\n(?=[{vietnamese_lower}])', ' ', text)
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

# --- 2. HÀM XỬ LÝ CHÍNH: QUÉT MỌI FILE TRONG THƯ MỤC ---
def process_and_ingest():
    if not os.path.exists(CRAWL_DIR):
        print(f"[LỖI] Không thấy thư mục: {CRAWL_DIR}")
        return

    documents = []
    metadatas = []
    
    file_list = [f for f in os.listdir(CRAWL_DIR) if f.lower().endswith(('.pdf', '.json'))]
    print(f"[*] Tìm thấy {len(file_list)} file phù hợp trong thư mục Crawl.")

    # Khởi tạo Splitter giống hệt code gốc của bạn
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""]
    )

    for file_name in file_list:
        file_path = os.path.join(CRAWL_DIR, file_name)
        
        # A. NẾU LÀ FILE PDF: Dùng logic trích xuất block cao cấp của bạn
        if file_name.lower().endswith(".pdf"):
            print(f"   --> Đang xử lý PDF: {file_name}")
            try:
                doc = fitz.open(file_path)
                for page_num, page in enumerate(doc):
                    # Lấy text theo blocks để giữ thứ tự đọc tự nhiên
                    blocks = page.get_text("blocks")
                    page_content = "\n".join([b[4] for b in blocks if b[6] == 0])
                    cleaned_content = clean_medical_text(page_content)
                    
                    if len(cleaned_content) > 50:
                        chunks = text_splitter.split_text(cleaned_content)
                        for chunk in chunks:
                            if len(chunk) > 50:
                                documents.append(chunk)
                                metadatas.append({"source": file_name, "page": page_num + 1})
                doc.close()
            except Exception as e:
                print(f"      [Lỗi PDF] {e}")

        # B. NẾU LÀ FILE JSON: Đọc tiêu đề + nội dung
        elif file_name.lower().endswith(".json"):
            print(f"   --> Đang xử lý JSON: {file_name}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        full_text = f"Tiêu đề: {item.get('title', '')}\nNội dung: {item.get('content', '')}"
                        cleaned_json = clean_medical_text(full_text)
                        if len(cleaned_json) > 50:
                            chunks = text_splitter.split_text(cleaned_json)
                            for chunk in chunks:
                                documents.append(chunk)
                                metadatas.append({"source": file_name, "type": "crawl_data"})
            except Exception as e:
                print(f"      [Lỗi JSON] {e}")

    # --- 3. LƯU VÀO VECTOR DB (Giữ nguyên cấu hình CPU & Batching của bạn) ---
    if not documents:
        print("[-] Không có dữ liệu hợp lệ để nạp.")
        return

    print(f"\n[*] Đã chuẩn bị {len(documents)} chunks. Đang khởi động Embedding Model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except:
        pass
        
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 64
    ids = [f"doc_{i}" for i in range(len(documents))]
    
    for i in tqdm(range(0, len(documents), batch_size), desc="Mã hóa & Nạp DB"):
        end_idx = i + batch_size
        batch_docs = documents[i:end_idx]
        batch_metas = metadatas[i:end_idx]
        batch_ids = ids[i:end_idx]
        
        batch_embeddings = embedder.encode(batch_docs, show_progress_bar=False).tolist()
        
        collection.add(
            documents=batch_docs,
            embeddings=batch_embeddings,
            metadatas=batch_metas,
            ids=batch_ids
        )

    print(f"\n[🚀 THÀNH CÔNG] Đã nạp {len(documents)} chunks từ toàn bộ thư mục Crawl vào {DB_PATH}.")

if __name__ == "__main__":
    process_and_ingest()