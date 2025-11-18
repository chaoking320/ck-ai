from fastapi import FastAPI
from pydantic import BaseModel
from ai_service_demo import ChatService

app = FastAPI()
chat_service = ChatService("qwen3:4b")

'''
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{"message": "你好，介绍一下你自己"}'
'''

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    response = chat_service.chat(request.message, request.history)
    return {"choices": [{"message": {"content": response, "role": "assistant"}}]}

# 添加这部分以便能够启动服务器
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)