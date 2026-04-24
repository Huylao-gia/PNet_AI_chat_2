import json
import os
import logging

# Thiết lập Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LlamaDataFormatter:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Cố định System Prompt cho toàn bộ quá trình train để model quen vai trò
        self.system_prompt = (
            "Bạn là một chuyên gia thú y AI tận tâm, thấu cảm và có chuyên môn cao. "
            "Nhiệm vụ của bạn là đọc kỹ NGỮ CẢNH được cung cấp và trả lời câu hỏi của người dùng. "
            "TUYỆT ĐỐI tuân thủ các quy tắc an toàn y tế: không kê đơn thuốc online, và luôn khuyên "
            "đưa thú cưng đi khám nếu có dấu hiệu nguy hiểm."
        )

    def _format_single_qa(self, qa_pair: dict) -> dict:
        """Chuyển đổi 1 cặp QA thô thành định dạng messages."""
        original_content = qa_pair.get("original_content", "").strip()
        question = qa_pair.get("question", "").strip()
        answer = qa_pair.get("answer", "").strip()
        
        # Cấu trúc RAG: Gắn context vào User Prompt
        user_content = f"NGỮ CẢNH:\n{original_content}\n\nCÂU HỎI:\n{question}"
        
        return {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer}
            ]
        }

    def process_split(self, split_name: str):
        """Xử lý và lưu file JSONL cho từng tập dữ liệu (train/eval/test)."""
        input_file = os.path.join(self.input_dir, f"raw_qa_{split_name}.json")
        output_file = os.path.join(self.output_dir, f"{split_name}_ready.jsonl")
        
        if not os.path.exists(input_file):
            logger.warning(f"Không tìm thấy file {input_file}. Bỏ qua...")
            return

        logger.info(f"Đang xử lý tập {split_name.upper()}...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        formatted_data = [self._format_single_qa(qa) for qa in data]
        
        # Ghi ra định dạng JSON Lines (.jsonl)
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in formatted_data:
                # Đảm bảo tiếng Việt không bị lỗi font khi dump json
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')
                
        logger.info(f"✅ Đã tạo file {output_file} ({len(formatted_data)} mẫu).")

    def run(self):
        # Chạy format cho cả 3 tập
        splits = ["train", "eval", "test"]
        for split in splits:
            self.process_split(split)
        
        logger.info("🎉 HOÀN TẤT CHUẨN BỊ DỮ LIỆU HUẤN LUYỆN!")

if __name__ == "__main__":
    # Đường dẫn tương ứng với code sinh dữ liệu ở bước 2
    INPUT_DIR = "data/distillation"
    OUTPUT_DIR = "data/ready"
    
    formatter = LlamaDataFormatter(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)
    formatter.run()
