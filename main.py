from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from google import genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key="")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    keywords: List[str]
    messages: List[Message]

class ChatResponse(BaseModel):
    reply: str

def build_system_prompt(keywords: List[str]) -> str:
    keyword_str = ", ".join(keywords)
    return f"""당신은 사용자의 성향을 파악하는 AI입니다.
사용자가 성향 테스트에서 선택한 키워드: {keyword_str}

역할 지침:
- 사용자와 편하게 대화하면서 성격, 관심사, 가치관을 자연스럽게 파악하세요
- 딱딱한 질문지처럼 묻지 말고 친구처럼 자연스럽게 대화하세요
- 한 번에 한 가지만 물어보세요
- 2~3문장 이내로 짧게 말하세요
- 한국어로 대화하세요
- 앞서 선택한 키워드를 참고해서 관련된 주제로 대화를 이끌어 나가세요"""

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        system_prompt = build_system_prompt(request.keywords)

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
            model="gemini-2.0-flash-lite",
            contents=history + [genai.types.Content(
                role="user",
                parts=[genai.types.Part(text=last_message)]
            )],
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
            )
        )

        return ChatResponse(reply=response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
