import json
import os
import asyncio
import logging
from typing import Dict, List
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# Load biến môi trường
load_dotenv()

# Thiết lập Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorpusProofreader:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        
        api_key = os.getenv("OPENAI_API_KEY", "None")
        if not api_key or api_key == "None":
            raise ValueError("Vui lòng thiết lập OPENAI_API_KEY trong file .env")
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.model_name = "gpt-4o-mini"
        
        # Giới hạn luồng đồng thời để tránh Rate Limit
        self.semaphore = asyncio.Semaphore(10)
        self.max_retries = 3

    def _get_system_prompt(self) -> str:
        return """Bạn là một biên tập viên Tiếng Việt xuất sắc và tỉ mỉ. 
Nhiệm vụ của bạn là nhận vào Title và Content, sau đó sửa toàn bộ các lỗi:
1. Thiếu dấu câu, sai lỗi chính tả.
2. Từ viết tắt không chuẩn, khoảng trắng thừa hoặc thiếu.
3. Lỗi viết hoa, viết thường không đúng quy tắc.

YÊU CẦU BẮT BUỘC (KHÔNG ĐƯỢC VI PHẠM):
- GIỮ NGUYÊN 100% ý nghĩa, độ dài và văn phong của bản gốc. 
- TUYỆT ĐỐI KHÔNG tự ý thêm thắt ý tưởng, KHÔNG tóm tắt, KHÔNG diễn đạt lại câu (paraphrase).
- Chỉ sửa lỗi hình thức. Nếu văn bản đã chuẩn, hãy giữ nguyên.

Trả về kết quả dưới định dạng JSON chính xác:
{"title": "title đã sửa", "content": "content đã sửa"}"""

    async def _proofread_chunk(self, chunk: Dict) -> Dict:
        """Gọi API để sửa lỗi chính tả cho 1 bài viết."""
        async with self.semaphore:
            original_title = chunk.get("title", "")
            original_content = chunk.get("content", "")
            
            # Bỏ qua nếu content quá ngắn hoặc rỗng
            if not original_content or len(original_content) < 20:
                return chunk

            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": f"TITLE:\n{original_title}\n\nCONTENT:\n{original_content}"}
            ]
            
            base_delay = 2
            for attempt in range(self.max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        response_format={ "type": "json_object" },
                        messages=messages,
                        temperature=0.1, # Đặt temperature rất thấp để tránh model sáng tạo
                    )
                    
                    result = json.loads(response.choices[0].message.content)
                    
                    # Cập nhật lại data đã sửa
                    chunk["title"] = result.get("title", original_title)
                    chunk["content"] = result.get("content", original_content)
                    chunk["is_proofread"] = True # Gắn cờ để biết đã xử lý
                    return chunk
                    
                except Exception as e:
                    error_str = str(e).lower()
                    if "429" in error_str or "rate limit" in error_str:
                        wait_time = base_delay * (2 ** attempt)
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Lỗi khi xử lý URL {chunk.get('url')}: {e}")
                        if attempt == self.max_retries - 1:
                            return chunk # Nếu lỗi vẫn trả về chunk gốc để không mất data
                        await asyncio.sleep(base_delay)
            return chunk

    async def run(self):
        logger.info(f"Đọc dữ liệu từ {self.input_file}...")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Bắt đầu sửa lỗi chính tả cho {len(data)} bài viết...")
        
        # Lên lịch cho tất cả các tasks
        tasks = [self._proofread_chunk(chunk) for chunk in data]
        
        # Chạy đồng thời với thanh tiến trình
        cleaned_data = await tqdm.gather(*tasks, desc="Đang Proofread Corpus")
        
        # Lưu file
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
            
        logger.info(f"✅ Hoàn tất! Đã lưu file sạch tại: {self.output_file}")

async def main():
    # Điều chỉnh đường dẫn file theo project
    INPUT_PATH = "data/processed/cleaned_corpus.json"
    OUTPUT_PATH = "data/processed/proofread_corpus.json"
    
    proofreader = CorpusProofreader(input_file=INPUT_PATH, output_file=OUTPUT_PATH)
    await proofreader.run()

if __name__ == "__main__":
    asyncio.run(main())
