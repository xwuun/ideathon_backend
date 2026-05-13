from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from google import genai
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    reply: str

SYSTEM_PROMPT = """당신은 사용자의 성격을 파악하는 AI입니다.

역할 지침:
- 대화를 통해 사용자의 성격, 가치관, 관심사를 자연스럽게 파악하세요
- 총 3~5개의 질문을 대화 흐름에 맞게 자연스럽게 해주세요
- 한 번에 한 가지 질문만 하세요
- 친구에게 말하듯 편하고 자연스럽게 대화하세요
- 2~3문장 이내로 짧게 말하세요
- 한국어로 대화하세요
- 첫 메시지에서는 반갑게 인사하고 첫 번째 질문을 해주세요"""

@app.post("/chat/start", response_model=ChatResponse)
async def chat_start():
    try:
        response = client.models.generate_content(
            model="models/gemini-2.0-flash",
            contents=[genai.types.Content(
                role="user",
                parts=[genai.types.Part(text="대화를 시작해줘.")]
            )],
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
            )
        )
        return ChatResponse(reply=response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        history = []
        for msg in request.messages[:-1]:
            history.append(
                genai.types.Content(
                    role=msg.role,
                    parts=[genai.types.Part(text=msg.content)]
                )
            )

        last_message = request.messages[-1].content

        response = client.models.generate_content(
            model="models/gemini-2.5-flash-preview-05-20",
            contents=history + [genai.types.Content(
                role="user",
                parts=[genai.types.Part(text=last_message)]
            )],
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
            )
        )
        return ChatResponse(reply=response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
