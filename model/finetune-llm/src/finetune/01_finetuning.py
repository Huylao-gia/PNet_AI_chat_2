# %% [markdown]
# # GIAI ĐOẠN 3: HUẤN LUYỆN LLAMA-3.2-1B TỐI ƯU CHO RAG VỚI UNSLOTH

# %% [markdown]
# ### BƯỚC 1: Cài đặt thư viện
# %%
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers trl peft accelerate bitsandbytes
# !pip install wandb datasets

# %%
import torch
import wandb
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments

# Điền API Key của WandB (nếu có)
# wandb.login(key="YOUR_WANDB_KEY")

# %% [markdown]
# ### BƯỚC 2: Nạp Mô hình gốc chuẩn Meta & Cấu hình Context dài
# %%
max_seq_length = 8192 # Nâng lên 8192 để đọc context RAG thoải mái
dtype = None 
load_in_4bit = True 

print("Đang nạp mô hình Base Llama-3.2-1B-Instruct...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct", # BẮT BUỘC DÙNG MODEL GỐC NÀY
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Cấu hình Tokenizer nhận diện định dạng Llama-3 chuẩn
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1", # Llama 3.2 dùng chung template với 3.1
)

# %% [markdown]
# ### BƯỚC 3: Cấu hình QLoRA Adapters
# %%
model = FastLanguageModel.get_peft_model(
    model,
    r = 32, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"], 
    lora_alpha = 64,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 42,
    use_rslora = False, 
)

# %% [markdown]
# ### BƯỚC 4: Nạp và Định dạng Dữ liệu (Train & Eval)
# Đọc file dữ liệu đã chia sẵn để chống Data Leakage.
# %%
# Đảm bảo bạn đã upload train_ready.jsonl và eval_ready.jsonl lên Colab
train_dataset = load_dataset("json", data_files="train_ready.jsonl", split="train")
eval_dataset = load_dataset("json", data_files="eval_ready.jsonl", split="train")

# Hàm chuyển đổi mảng 'messages' thành chuỗi ChatML chuẩn
def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return { "text" : texts }

# Áp dụng cho cả 2 tập dữ liệu
train_dataset = train_dataset.map(formatting_prompts_func, batched = True)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched = True)

print(f"Số mẫu Train: {len(train_dataset)} | Số mẫu Eval: {len(eval_dataset)}")

# %% [markdown]
# ### BƯỚC 5: Thiết lập Trainer với Evaluation Strategy
# %%
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset, # Đưa tập Eval vào để chấm điểm
    dataset_text_field = "text", 
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, 
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, 
        warmup_steps = 50, 
        num_train_epochs = 3, # Tăng lên 3 epochs để mô hình ngấm sâu kiến thức
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        eval_strategy = "steps", # Đánh giá sau mỗi X bước
        eval_steps = 50, # Cứ 50 bước train thì lấy tập eval ra test
        save_strategy = "steps",
        save_steps = 50, # Lưu checkpoint cùng lúc với eval
        optim = "adamw_8bit", 
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "outputs",
        report_to = "wandb", 
        load_best_model_at_end = True, # Giữ lại checkpoint có Validation Loss thấp nhất
    ),
)

# %% [markdown]
# ### BƯỚC 6: Bắt đầu Huấn luyện 🚀
# Theo dõi biểu đồ trên WandB. Nếu Train Loss giảm nhưng Eval Loss tăng, mô hình đang bị Overfit.
# %%
trainer_stats = trainer.train()

# %% [markdown]
# ### BƯỚC 7: Kiểm thử trực tiếp (Inference Test chuẩn RAG)
# %%
FastLanguageModel.for_inference(model) 

system_prompt = "Bạn là trợ lý AI chuyên gia thú y tận tâm. Hãy trả lời câu hỏi dựa trên thông tin tham khảo được cung cấp một cách chính xác và đồng cảm. Tuyệt đối không tự bịa đặt."
context = "Bệnh Care ở chó (Canine Distemper) là bệnh truyền nhiễm nguy hiểm do virus gây ra, ảnh hưởng đến hệ hô hấp, tiêu hóa và thần kinh. Chó chưa tiêm phòng rất dễ mắc bệnh."
question = "Chó nhà tôi chưa tiêm phòng, dạo này thấy ho và tiêu chảy. Liệu có phải bệnh Care không?"

# Sử dụng messages list thay vì ghép chuỗi thủ công
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"NGỮ CẢNH:\n{context}\n\nCÂU HỎI:\n{question}"}
]

# Tự động render prompt chuẩn
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Kích hoạt bot trả lời
    return_tensors = "pt",
).to("cuda")

# Sinh câu trả lời
outputs = model.generate(input_ids = inputs, max_new_tokens = 512, use_cache = True)
# Chỉ in ra phần token mới sinh ra
response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

print("\n--- CÂU TRẢ LỜI CỦA MÔ HÌNH ---")
print(response.strip())

# %% [markdown]
# ### BƯỚC 8: Xuất Mô hình (GGUF Export)
# %%
model.save_pretrained_gguf("Llama-3.2-1B-Pet-Chatbot-GGUF", tokenizer, quantization_method = "q4_k_m")
print("Đã lưu thành công thư mục: Llama-3.2-1B-Pet-Chatbot-GGUF")