import json
import os
import asyncio
import logging
import random
from typing import Dict, List, Any
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# Tải biến môi trường
load_dotenv()

# Thiết lập Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataDistillationPipeline:
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Cấu hình API
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ KHÔNG TÌM THẤY API KEY! Vui lòng kiểm tra file .env.")
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.model_name = "gpt-4o-mini"
        self.semaphore = asyncio.Semaphore(10) # Giới hạn 10 luồng đồng thời
        self.max_retries = 5

    def _split_dataset(self, data: List[Dict]) -> Dict[str, List[Dict]]:
        """Chia tách dataset dựa trên bài viết gốc để chống Rò rỉ dữ liệu (Data Leakage)"""
        logger.info("Đang chia tách dataset (Train 80% / Eval 10% / Test 10%)...")
        # Xáo trộn dữ liệu gốc với seed cố định để đảm bảo tính nhất quán nếu chạy lại
        random.seed(42)
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        
        n = len(shuffled_data)
        train_end = int(n * 0.8)
        eval_end = int(n * 0.9)
        
        splits = {
            "train": shuffled_data[:train_end],
            "eval": shuffled_data[train_end:eval_end],
            "test": shuffled_data[eval_end:]
        }
        
        logger.info(f"Đã chia: Train ({len(splits['train'])} chunks), Eval ({len(splits['eval'])} chunks), Test ({len(splits['test'])} chunks).")
        return splits

    def _get_personas(self) -> Dict[str, Dict[str, Any]]:
        """Định nghĩa 5 Personas với Few-Shot Prompting để tạo QA chất lượng cao."""
        base_rules = """YÊU CẦU KHẮT KHE:
- Văn phong tự nhiên của người Việt Nam.
- BẮT BUỘC KHÔNG sử dụng các cụm từ: "Theo bài viết", "Dựa vào đoạn văn", "Như đã đề cập", "Thông tin cho thấy".
- Trả về JSON chính xác: {"qa_pairs": [{"question": "...", "answer": "..."}]}"""

        return {
            "direct": {
                "num_qa": 3,
                "prompt": f"""Bạn là Bác sĩ Thú y AI. Dựa vào NỘI DUNG GỐC, tạo 3 cặp QA (Hỏi trực tiếp vào vấn đề).
{base_rules}
VÍ DỤ MẪU:
Q: Chó con bao nhiêu ngày tuổi thì có thể bắt đầu chích ngừa mũi đầu tiên?
A: Chào bạn, chó con có thể bắt đầu chích ngừa mũi vaccine đầu tiên khi được 6 tuần tuổi (khoảng 42-45 ngày). Trước đó bạn nhớ xổ giun cho bé nhé!"""
            },
            "symptom": {
                "num_qa": 3,
                "prompt": f"""Bạn là Bác sĩ Thú y AI. Dựa vào NỘI DUNG GỐC, tạo 3 cặp QA (Người dùng mô tả triệu chứng, không biết tên bệnh).
{base_rules}
VÍ DỤ MẪU:
Q: Mèo nhà em tự nhiên bỏ ăn 2 hôm nay, nôn ra bọt trắng và nằm co ro ở góc nhà, bé bị sao vậy bác sĩ?
A: Chào bạn, triệu chứng bỏ ăn, nôn bọt trắng và nằm co ro có thể là dấu hiệu bé bị đau bụng, nguy cơ cao là nhiễm virus Giảm bạch cầu (FPV) hoặc viêm ruột. Đây là tình trạng nguy hiểm, bạn cần giữ ấm và đưa bé đến phòng khám thú y để test nhanh ngay lập tức nhé."""
            },
            "panic": {
                "num_qa": 2,
                "prompt": f"""Bạn là Bác sĩ Thú y AI. Dựa vào NỘI DUNG GỐC, tạo 2 cặp QA (Người dùng hoảng loạn, viết tắt, lo lắng). AI cần trấn an và khuyên đi cấp cứu.
{base_rules}
VÍ DỤ MẪU:
Q: bs ơi cứu cún nhà e vs!!! nó co giật sùi bọt mép e sợ quá k biết làm s bjo 😭😭
A: Bình tĩnh bạn nhé! Triệu chứng co giật sùi bọt mép rất nguy hiểm, có thể do ngộ độc hoặc động kinh. Bạn tuyệt đối không vắt chanh hay đổ sữa vào miệng bé lúc này để tránh sặc vào phổi. Hãy dùng vật mềm kê đầu bé, quay video lại cơn giật và mang bé đến trạm thú y cấp cứu ngay lập tức!"""
            },
            "negative": {
                "num_qa": 2,
                "prompt": f"""Bạn là Bác sĩ Thú y AI. Dựa vào NỘI DUNG GỐC, tạo 2 cặp QA (Người dùng đòi kê đơn thuốc hoặc hỏi chữa mẹo). AI phải TỪ CHỐI kê đơn online, cảnh báo rủi ro và khuyên đi khám.
{base_rules}
VÍ DỤ MẪU:
Q: Bác sĩ ơi kê cho em liều kháng sinh trị tiêu chảy cho chó poodle 5kg với ạ, em tự ra tiệm thuốc tây mua cho nhanh.
A: Chào bạn, mình không thể kê đơn thuốc kháng sinh online cho bé được. Việc tự ý dùng thuốc tây của người cho chó có thể gây ngộ độc gan, suy thận hoặc sốc phản vệ dẫn đến tử vong. Bạn hãy mang bé ra trạm thú y để bác sĩ kiểm tra và kê đúng loại thuốc dành riêng cho thú cưng nhé."""
            },
            "multiturn": {
                "num_qa": 2,
                "prompt": f"""Bạn là Bác sĩ Thú y AI. Dựa vào NỘI DUNG GỐC, tạo 2 cặp QA (Câu hỏi mang tính chất tiếp nối, đào sâu chi phí, cách phòng ngừa, hoặc xét nghiệm cần làm).
{base_rules}
VÍ DỤ MẪU:
Q: Vậy nếu đưa bé đi khám nghi bị Care thì thường bác sĩ sẽ làm những xét nghiệm gì ạ?
A: Khi đưa bé đến phòng khám, đầu tiên bác sĩ sẽ lấy dịch mắt hoặc dịch mũi để làm Test nhanh (Test kit) kiểm tra virus Care. Nếu cần thiết, bác sĩ có thể chỉ định xét nghiệm thêm máu để đánh giá mức độ viêm nhiễm và tình trạng sức khỏe tổng thể của bé."""
            }
        }

    async def _generate_for_chunk(self, chunk: Dict, persona_name: str, persona_config: Dict, split_name: str) -> List[Dict]:
        """Gọi API sinh câu hỏi cho 1 bài viết theo 1 Persona."""
        async with self.semaphore:
            messages = [
                {"role": "system", "content": persona_config["prompt"]},
                {"role": "user", "content": f"NỘI DUNG GỐC:\n{chunk['content']}"}
            ]
            
            base_delay = 2
            for attempt in range(self.max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        response_format={ "type": "json_object" },
                        messages=messages,
                        temperature=0.7,
                    )
                    
                    result = json.loads(response.choices[0].message.content)
                    qa_pairs = result.get("qa_pairs", [])
                    
                    # Gắn metadata quan trọng để truy xuất sau này
                    for qa in qa_pairs:
                        qa["source_url"] = chunk.get("url", "unknown")
                        qa["persona"] = persona_name
                        qa["split"] = split_name
                        qa["original_content"] = chunk["content"]
                        
                    return qa_pairs
                    
                except Exception as e:
                    error_str = str(e).lower()
                    if "429" in error_str or "rate limit" in error_str:
                        wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Lỗi tạo QA (Persona: {persona_name}): {e}")
                        if attempt == self.max_retries - 1:
                            return []
                        await asyncio.sleep(base_delay)
            return []

    async def run(self):
        logger.info(f"Đọc dữ liệu đã chuẩn hóa từ {self.input_file}...")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. Chia tách dữ liệu
        splits = self._split_dataset(data)
        personas = self._get_personas()
        
        # 2. Tạo Tasks cho từng tập dữ liệu (Train/Eval/Test)
        all_qa_results = {"train": [], "eval": [], "test": []}
        
        for split_name, chunks in splits.items():
            logger.info(f"Đang sinh QA cho tập: {split_name.upper()}...")
            tasks = []
            for chunk in chunks:
                for p_name, p_config in personas.items():
                    tasks.append(self._generate_for_chunk(chunk, p_name, p_config, split_name))
            
            # Chạy đồng thời các tasks của tập hiện tại
            split_results = await tqdm.gather(*tasks, desc=f"Tiến độ tập {split_name}")
            
            # Flatten list
            flat_results = [item for sublist in split_results for item in sublist]
            all_qa_results[split_name] = flat_results
            
            # Lưu riêng từng tập
            output_path = os.path.join(self.output_dir, f"raw_qa_{split_name}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(flat_results, f, ensure_ascii=False, indent=4)
            logger.info(f"Đã lưu {len(flat_results)} QA vào {output_path}")

        total_qa = sum(len(qa_list) for qa_list in all_qa_results.values())
        logger.info(f"✅ Hoàn tất Giai đoạn Distillation! Tổng cộng đã sinh ra {total_qa} cặp QA thô.")

async def main():
    INPUT_PATH = "data/processed/proofread_corpus.json"
    OUTPUT_DIR = "data/distillation"
    
    pipeline = DataDistillationPipeline(input_file=INPUT_PATH, output_dir=OUTPUT_DIR)
    await pipeline.run()

if __name__ == "__main__":
    asyncio.run(main())
