import os
import uuid
from typing import Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from .schemas import ChatRequest, ChatResponse

load_dotenv()

APP_NAME = os.getenv("APP_NAME", "fastapi-chatbot")
allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in allowed_origins else allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- serve static files (css/js) and an index page ---
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("templates/index.html")

# --- in-memory session store ---
History = List[Dict[str, str]]    # [{"role":"user"/"bot","content":"..."}]
SESSIONS: Dict[str, History] = {}

# --- tiny rule-based brain (starter) ---
def small_talk_brain(message: str) -> str:
    m = message.lower().strip()

    if not m:
        return "Say something and I’ll respond 🙂"

    greetings = ("hi", "hello", "hey", "yo")
    if any(m.startswith(g) for g in greetings):
        return "Hey! I’m your FastAPI bot. Ask me anything or say 'help'."

    if "help" in m:
        return (
            "I support:\n"
            "• Chat over HTTP: POST /chat\n"
            "• WebSocket chat: /ws (send JSON {message, session_id})\n"
            "I keep simple per-session history in memory."
        )

    if "fastapi" in m:
        return "FastAPI is a modern, fast web framework for building APIs with Python type hints."

    if "deploy" in m:
        return "We can Dockerize and deploy to Render/Railway/Fly—this project already includes a Dockerfile."

    if "clear" in m and "history" in m:
        return "If you send 'clear history', I’ll start fresh next turn."

    return f"You said: “{message}”. (Tip: try 'help' to see what I can do.)"

def get_or_create_session(session_id: str | None) -> str:
    if session_id and session_id in SESSIONS:
        return session_id
    new_id = str(uuid.uuid4())
    SESSIONS[new_id] = []
    return new_id

@app.get("/health")
def health():
    return {"status": "ok", "app": APP_NAME}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sid = get_or_create_session(req.session_id)
    user_msg = req.message.strip()

    if user_msg.lower() == "clear history":
        SESSIONS[sid] = []
        reply = "History cleared for this session. How can I help now?"
        SESSIONS[sid].append({"role": "bot", "content": reply})
        return ChatResponse(reply=reply, session_id=sid)

    SESSIONS[sid].append({"role": "user", "content": user_msg})
    reply = small_talk_brain(user_msg)
    SESSIONS[sid].append({"role": "bot", "content": reply})
    return ChatResponse(reply=reply, session_id=sid)

# --- Optional WebSocket for realtime chat ---
class WSMessage(BaseModel):
    message: str
    session_id: str | None = None

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            try:
                msg = WSMessage(**data)
            except Exception:
                await ws.send_json({"error": "Send JSON with 'message' and optional 'session_id'."})
                continue

            sid = get_or_create_session(msg.session_id)
            text = (msg.message or "").strip()
            if not text:
                await ws.send_json({"reply": "Say something and I’ll respond 🙂", "session_id": sid})
                continue

            if text.lower() == "clear history":
                SESSIONS[sid] = []
                await ws.send_json({"reply": "History cleared. What’s next?", "session_id": sid})
                continue

            SESSIONS[sid].append({"role": "user", "content": text})
            reply = small_talk_brain(text)
            SESSIONS[sid].append({"role": "bot", "content": reply})
            await ws.send_json({"reply": reply, "session_id": sid})
    except WebSocketDisconnect:
        return
