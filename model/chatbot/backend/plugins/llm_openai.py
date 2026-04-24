from openai import AsyncOpenAI
from typing import List, Dict, AsyncGenerator
from core.interfaces import BaseLLM

class OpenAIPlugin(BaseLLM):
    """Implement LLM sử dụng API của OpenAI (Đóng vai trò tạm thế cho vLLM)."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        if not api_key:
            raise ValueError("[Plugin] Thiếu OPENAI_API_KEY trong cấu hình!")
            
        print(f"[Plugin] Đang khởi tạo OpenAI Client với model '{model}'...")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def stream_chat(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=0.1 # Nhiệt độ thấp = Ưu tiên sự thật, bám sát context
        )
        
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
