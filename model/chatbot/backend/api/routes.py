import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from schemas.models import ChatRequest, ContextDocument
from services.rag_engine import RAGEngine
from core.interfaces import BaseVectorDB, BaseLLM, BaseMemory

router = APIRouter()

@router.post("/api/chat")
async def chat_endpoint(request: Request, body: ChatRequest):
    vectordb: BaseVectorDB = request.app.state.vectordb
    llm: BaseLLM = request.app.state.llm
    memory: BaseMemory = request.app.state.memory
    
    session_id = body.session_id
    user_query = body.message
    
    print("\n" + "="*60)
    print(f"🚀 [TRACE START] REQUEST MỚI - Session: {session_id}")
    print(f"👤 Câu hỏi (User Query): '{user_query}'")
    print("="*60)
    
    # ---------------------------------------------------------
    # STEP 1: TRUY VẤN VECTOR DB
    # ---------------------------------------------------------
    print("\n[STEP 1] 🔍 Đang truy vấn ChromaDB...")
    raw_contexts = vectordb.search(user_query, top_k=body.top_k)
    print(f"   -> Tìm thấy {len(raw_contexts)} chunks thô từ Database.")
    
    contexts = []
    for i, ctx in enumerate(raw_contexts):
        mapped_ctx = ContextDocument(
            page=str(ctx.get("page", "N/A")), 
            content=ctx.get("content", ""),
            confidence_score=ctx.get("score", 0.0) 
        )
        contexts.append(mapped_ctx)
        
        # LOG CHI TIẾT KẾT QUẢ TÌM KIẾM
        print(f"   📄 [Chunk {i+1}] Trang: {mapped_ctx.page} | Độ tự tin: {mapped_ctx.confidence_score}%")
        print(f"      Nội dung: {mapped_ctx.content[:200]}...\n") # In 200 ký tự đầu để bạn đọc thử
    
    # ---------------------------------------------------------
    # STEP 2: LẤY LỊCH SỬ HỘI THOẠI
    # ---------------------------------------------------------
    print("[STEP 2] 🧠 Đang tải lịch sử hội thoại (Memory)...")
    history = memory.get_history(session_id)
    print(f"   -> Có {len(history)} tin nhắn cũ trong session này.")
    
    # ---------------------------------------------------------
    # STEP 3: LẮP RÁP PROMPT
    # ---------------------------------------------------------
    print("\n[STEP 3] 🛠️ Đang lắp ráp RAG Prompt...")
    messages = RAGEngine.build_messages(contexts, history, user_query)
    
    # Trích xuất riêng System Prompt ra để Log xem nó có thực sự chứa Context không
    system_prompt = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
    print("   -> BẢN XEM TRƯỚC SYSTEM PROMPT (Dữ liệu thực tế gửi cho AI):")
    print("-" * 40)
    print(f"{system_prompt[:500]}...\n[...ĐÃ CẮT BỚT ĐỂ DỄ NHÌN...]") 
    print("-" * 40)
    
    # ---------------------------------------------------------
    # STEP 4: GỌI LLM VÀ STREAMING
    # ---------------------------------------------------------
    print("\n[STEP 4] 🤖 Đang stream câu trả lời từ LLM...")
    async def event_generator():
        full_ai_response = ""
        try:
            async for token in llm.stream_chat(messages):
                full_ai_response += token
                payload = json.dumps({"content": token}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
        finally:
            print(f"\n[TRACE END] 🏁 LLM đã stream xong. Độ dài câu trả lời: {len(full_ai_response)} ký tự.")
            print("="*60 + "\n")
            
            memory.add_message(session_id, "user", user_query)
            memory.add_message(session_id, "assistant", full_ai_response)
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
