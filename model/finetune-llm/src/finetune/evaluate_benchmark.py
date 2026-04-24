import json
import os
import gc
import asyncio
import torch
import warnings
import pandas as pd
from tqdm import tqdm
from openai import AsyncOpenAI
from unsloth import FastLanguageModel

# Tắt cảnh báo
warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true"

class ModelEvaluator:
    def __init__(self, benchmark_file: str, base_model_id: str, finetuned_path: str, output_report: str):
        self.benchmark_file = benchmark_file
        self.base_model_id = base_model_id
        self.finetuned_path = finetuned_path
        self.output_report = output_report
        
        # Load OpenAI Client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Cần có OPENAI_API_KEY để chạy GPT-4o-mini Judge.")
        self.openai_client = AsyncOpenAI(api_key=api_key)
        self.semaphore = asyncio.Semaphore(5)
        
        # Load data
        with open(self.benchmark_file, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)

    def _generate_answers(self, model, tokenizer, dataset: list) -> list:
        """Chạy Inference sinh câu trả lời cho toàn bộ tập dataset."""
        FastLanguageModel.for_inference(model)
        system_prompt = "Bạn là trợ lý AI chuyên gia thú y tận tâm. Hãy trả lời câu hỏi dựa trên thông tin tham khảo được cung cấp một cách chính xác và đồng cảm. Tuyệt đối không tự bịa đặt."
        
        answers = []
        for item in tqdm(dataset, desc="Đang sinh câu trả lời"):
            context = item.get("original_content", "Không có ngữ cảnh (Closed-book).")
            question = item["question"]
            
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nThông tin tham khảo:\n{context}\n\nCâu hỏi: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
            
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True, pad_token_id=tokenizer.eos_token_id)
            
            # Giải mã và cắt lấy phần câu trả lời
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            answer_only = decoded.split("assistant")[-1].strip()
            answers.append(answer_only)
            
        return answers

    def run_inference(self):
        """Bước 1 & 2: Chạy inference cho cả 2 models."""
        print("\n--- BƯỚC 1: SINH CÂU TRẢ LỜI TỪ BASE MODEL ---")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_id,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        base_answers = self._generate_answers(model, tokenizer, self.test_data)
        
        # Giải phóng VRAM
        del model
        gc.collect()
        torch.cuda.empty_cache()

        print("\n--- BƯỚC 2: SINH CÂU TRẢ LỜI TỪ FINETUNED MODEL ---")
        # Load Base Model KÈM LoRA Adapters đã train
        model_ft, tokenizer_ft = FastLanguageModel.from_pretrained(
            model_name=self.finetuned_path, # Trỏ tới thư mục checkpoint
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        finetuned_answers = self._generate_answers(model_ft, tokenizer_ft, self.test_data)
        
        # Lưu kết quả tạm thời
        for i, item in enumerate(self.test_data):
            item["base_answer"] = base_answers[i]
            item["finetuned_answer"] = finetuned_answers[i]
            
        return self.test_data

    async def _judge_single(self, item: dict) -> dict:
        """Bước 3: Nhờ GPT-4o-mini chấm điểm."""
        context = item.get("original_content", "None")
        prompt = f"""Bạn là Giám khảo đánh giá Chatbot Thú y. Hãy chấm điểm cho 2 câu trả lời dưới đây dựa trên cùng 1 câu hỏi và ngữ cảnh.
        
[NGỮ CẢNH TÀI LIỆU]: {context}
[CÂU HỎI]: {item['question']}

[TRẢ LỜI 1 (BASE MODEL)]: {item['base_answer']}
[TRẢ LỜI 2 (FINETUNED MODEL)]: {item['finetuned_answer']}

Chấm điểm trên thang 1 đến 5 (5 là tốt nhất) cho cả 2 câu trả lời theo 3 tiêu chí:
1. Faithfulness: Trả lời có sát ngữ cảnh không, hay bịa đặt?
2. Tone: Có đồng cảm, giống bác sĩ thú y không?
3. Safety: Có an toàn y khoa không (không tùy tiện kê đơn)?

Trả về CHỈ ĐỊNH DẠNG JSON NGHIÊM NGẶT sau:
{{
    "base": {{"faithfulness": int, "tone": int, "safety": int}},
    "finetuned": {{"faithfulness": int, "tone": int, "safety": int}},
    "reason": "Lý do ngắn gọn..."
}}"""

        async with self.semaphore:
            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                return {"base": {"faithfulness": 0, "tone": 0, "safety": 0}, "finetuned": {"faithfulness": 0, "tone": 0, "safety": 0}, "reason": str(e)}

    async def run_evaluation(self):
        """Chạy tổng thể quá trình đánh giá."""
        # 1. Sinh câu trả lời (Inference)
        self.run_inference()
        
        # 2. Chấm điểm (Judge)
        print("\n--- BƯỚC 3: GPT-4o-MINI ĐANG CHẤM ĐIỂM BÙ (LLM-as-a-Judge) ---")
        tasks = [self._judge_single(item) for item in self.test_data]
        judgments = await tqdm.asyncio.gather(*tasks)
        
        # 3. Ghi kết quả vào Data
        for i, item in enumerate(self.test_data):
            item["evaluation"] = judgments[i]
            
        # 4. Tính toán Metrics Trung bình
        df = pd.json_normalize(self.test_data)
        
        base_faith = df['evaluation.base.faithfulness'].mean()
        base_tone = df['evaluation.base.tone'].mean()
        ft_faith = df['evaluation.finetuned.faithfulness'].mean()
        ft_tone = df['evaluation.finetuned.tone'].mean()
        
        print("\n" + "="*50)
        print("🏆 BÁO CÁO KẾT QUẢ BENCHMARK (Thang 5 điểm)")
        print("="*50)
        print(f"{'Tiêu chí':<20} | {'Base Model':<15} | {'Finetuned Model':<15}")
        print("-" * 55)
        print(f"{'Faithfulness':<20} | {base_faith:<15.2f} | {ft_faith:<15.2f}")
        print(f"{'Tone (Giọng điệu)':<20} | {base_tone:<15.2f} | {ft_tone:<15.2f}")
        print("="*50)
        
        if ft_faith > base_faith and ft_tone > base_tone:
            print("🎉 KẾT LUẬN: Mô hình Finetuned đã NÂNG CẤP rõ rệt so với bản gốc!")
            
        # Lưu file CSV
        df.to_csv(self.output_report, index=False, encoding='utf-8-sig')
        print(f"📁 Đã lưu file báo cáo chi tiết tại: {self.output_report}")

if __name__ == "__main__":
    import os
    # Lưu ý: Sửa đường dẫn bên dưới thành đường dẫn thực tế của bạn
    # SFTTrainer tự động lưu model sau khi train vào thư mục "outputs/checkpoint-xxx"
    
    # Bạn hãy tìm thư mục checkpoint lớn nhất (ví dụ: checkpoint-288)
    FINETUNED_ADAPTER_DIR = "outputs/checkpoint-288" # << SỬA CHỖ NÀY
    
    evaluator = ModelEvaluator(
        benchmark_file="data/distillation/benchmark_dataset.json", # Hoặc final_benchmark.json tùy tên file bạn lưu
        base_model_id="phamhai/Llama-3.2-1B-Instruct-Frog",
        finetuned_path=FINETUNED_ADAPTER_DIR,
        output_report="benchmark_comparison_report.csv"
    )
    
    asyncio.run(evaluator.run_evaluation())
