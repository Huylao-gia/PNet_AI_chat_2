import logging
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from core.interfaces import BaseVectorDB

# Tắt bớt log rác của thư viện
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

class ChromaDBPlugin(BaseVectorDB):
    """Implement ChromaDB sử dụng SBERT để mã hóa tiếng Việt."""
    
    def __init__(self, db_path: str, collection_name: str, model_name: str):
        print(f"[Plugin] Đang khởi tạo ChromaDB tại {db_path}...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)
        
        print(f"[Plugin] Đang nạp model Embedding '{model_name}'...")
        self.embedder = SentenceTransformer(model_name, device="cpu")
        
        # Khởi động nóng
        self.embedder.encode(["warm up"], show_progress_bar=False)
        print("[Plugin] VectorDB Plugin sẵn sàng!")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        # Chuyển câu hỏi thành vector
        query_vector = self.embedder.encode([query], show_progress_bar=False).tolist()
        
        # Truy vấn
        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=top_k
        )
        
        # Format kết quả chuẩn theo Interface
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        retrieved_data = []
        for i in range(len(documents)):
            confidence = round((1 - distances[i]) * 100, 1)
            retrieved_data.append({
                "content": documents[i],
                "page": metadatas[i].get("page", "N/A"),
                "score": confidence
            })
            
        return retrieved_data
