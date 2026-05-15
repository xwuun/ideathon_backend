from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from google import genai
import os
import random

# ─────────────────────────────────────────
# Groq는 openai 호환 클라이언트로 사용
# pip install groq
# ─────────────────────────────────────────
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# 여러 Gemini API 키 등록
# 환경변수: GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...
# 하나만 있으면 GEMINI_API_KEY 하나만 등록해도 됨
# ─────────────────────────────────────────
def load_gemini_keys() -> List[str]:
    keys = []
    # 단일 키 지원
    if os.environ.get("GEMINI_API_KEY"):
        keys.append(os.environ["GEMINI_API_KEY"])
    # 복수 키 지원: GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...
    i = 1
    while True:
        key = os.environ.get(f"GEMINI_API_KEY_{i}")
        if not key:
            break
        if key not in keys:
            keys.append(key)
        i += 1
    return keys

GEMINI_KEYS = load_gemini_keys()
gemini_key_index = 0  # 현재 사용 중인 키 인덱스

def get_next_gemini_client():
    """키를 순서대로 로테이션하며 반환"""
    global gemini_key_index
    if not GEMINI_KEYS:
        return None
    key = GEMINI_KEYS[gemini_key_index % len(GEMINI_KEYS)]
    gemini_key_index += 1
    return genai.Client(api_key=key)

# Groq 클라이언트 초기화
groq_client = None
if GROQ_AVAILABLE and os.environ.get("GROQ_API_KEY"):
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# ─────────────────────────────────────────
# 데이터 모델
# ─────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    reply: str
    provider: Optional[str] = None  # 어떤 API가 응답했는지 확인용 (선택)

# ─────────────────────────────────────────
# 시스템 프롬프트
# ─────────────────────────────────────────
SYSTEM_PROMPT = """당신은 사용자의 성격을 파악하는 AI입니다.
역할 지침:
- 반드시 한국어로만 대화하세요. 절대 다른 언어를 사용하지 마세요.
- 대화를 통해 사용자의 성격, 가치관, 관심사를 자연스럽게 파악하세요
- 총 3~5개의 질문을 대화 흐름에 맞게 자연스럽게 해주세요
- 한 번에 한 가지 질문만 하세요
- 친구에게 말하듯 편하고 자연스럽게 대화하세요
- 2~3문장 이내로 짧게 말하세요
- 한국어로 대화하세요
- 첫 메시지에서는 반갑게 인사하고 첫 번째 질문을 해주세요"""

# ─────────────────────────────────────────
# Groq 폴백 함수
# ─────────────────────────────────────────
def call_groq(messages: List[Message], system_prompt: str) -> str:
    if not groq_client:
        raise RuntimeError("Groq 클라이언트가 설정되지 않았습니다.")

    groq_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        # Groq는 role이 "user" / "assistant" 만 허용
        role = "assistant" if msg.role == "model" else msg.role
        groq_messages.append({"role": role, "content": msg.content})

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # 무료 한도가 넉넉한 모델
        messages=groq_messages,
        max_tokens=512,
    )
    return response.choices[0].message.content

# ─────────────────────────────────────────
# Gemini 호출 함수 (키 로테이션 + 자동 폴백)
# ─────────────────────────────────────────
def call_with_fallback(messages: List[Message]) -> tuple[str, str]:
    """
    1. Gemini 키들을 순서대로 시도
    2. 모두 한도 초과(429) 또는 실패 시 Groq로 폴백
    반환: (응답 텍스트, 사용된 프로바이더 이름)
    """
    last_error = None

    # Gemini 키 전부 시도
    for _ in range(max(len(GEMINI_KEYS), 1)):
        client = get_next_gemini_client()
        if not client:
            break
        try:
            history = []
            for msg in messages[:-1]:
                history.append(
                    genai.types.Content(
                        role=msg.role,
                        parts=[genai.types.Part(text=msg.content)]
                    )
                )
            last_message = messages[-1].content

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=history + [genai.types.Content(
                    role="user",
                    parts=[genai.types.Part(text=last_message)]
                )],
                config=genai.types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                )
            )
            return response.text, "gemini"

        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            # 429(한도 초과) 또는 quota 관련 에러면 다음 키 시도
            if "429" in error_str or "quota" in error_str or "rate" in error_str:
                continue
            # 다른 에러는 바로 Groq로 넘김
            break

    # Groq 폴백
    if groq_client:
        try:
            reply = call_groq(messages, SYSTEM_PROMPT)
            return reply, "groq"
        except Exception as groq_error:
            raise HTTPException(
                status_code=500,
                detail=f"Gemini 실패: {last_error} / Groq 실패: {groq_error}"
            )

    # 둘 다 실패
    raise HTTPException(status_code=500, detail=f"모든 API 실패: {last_error}")

# ─────────────────────────────────────────
# 엔드포인트
# ─────────────────────────────────────────
@app.post("/chat/start", response_model=ChatResponse)
async def chat_start():
    start_message = Message(role="user", content="대화를 시작해줘.")
    reply, provider = call_with_fallback([start_message])
    return ChatResponse(reply=reply, provider=provider)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="메시지가 없습니다.")
    reply, provider = call_with_fallback(request.messages)
    return ChatResponse(reply=reply, provider=provider)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gemini_keys": len(GEMINI_KEYS),
        "groq_available": groq_client is not None,
    }
