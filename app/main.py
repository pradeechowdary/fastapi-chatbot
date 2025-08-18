import os
import uuid
import time
import json
from typing import Dict, List, Optional
from collections import defaultdict
from ipaddress import ip_address

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from .schemas import ChatRequest, ChatResponse
from .rag_utils import RAGEngine, ensure_data_dir

load_dotenv()

APP_NAME   = os.getenv("APP_NAME", "Pradeep • Résumé Bot")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
RESUME_PATH  = os.getenv("RESUME_PATH", os.path.join("data", "PonnamcCV.pdf"))

DATA_DIR = ensure_data_dir()
LOG_PATH = os.path.join(DATA_DIR, "chat.log.jsonl")

app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# static & index
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("templates/index.html")

# sessions
History = List[Dict[str, str]]
SESSIONS: Dict[str, History] = {}

def get_or_create_session(session_id: Optional[str]) -> str:
    if session_id and session_id in SESSIONS:
        return session_id
    sid = str(uuid.uuid4())
    SESSIONS[sid] = []
    return sid

# ---- tiny IP rate limiter (burst 10, refill 1/sec) ----
TOKENS = defaultdict(lambda: {"tokens": 10, "ts": time.time()})
def rate_limit_ok(ip: str, cost: int = 1, cap: int = 10, refill_per_sec: float = 1.0) -> bool:
    b = TOKENS[ip]
    now = time.time()
    b["tokens"] = min(cap, b["tokens"] + (now - b["ts"]) * refill_per_sec)
    b["ts"] = now
    if b["tokens"] >= cost:
        b["tokens"] -= cost
        return True
    return False

def client_ip(req: Request) -> str:
    # basic best-effort
    xff = req.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return req.client.host if req.client else "0.0.0.0"

# ---- startup: find resume robustly, build index if missing ----
@app.on_event("startup")
def _startup():
    ensure_data_dir()
    try:
        RAGEngine.load()  # already built
        return
    except Exception:
        pass

    # 1) exact env path
    def exists(p: str) -> bool:
        return bool(p) and os.path.exists(p)

    chosen = None
    if exists(RESUME_PATH):
        chosen = RESUME_PATH
    else:
        # 2) default path
        cand = os.path.join("data", "PonnamcCV.pdf")
        if exists(cand):
            chosen = cand
        else:
            # 3) any pdf under data/
            for name in os.listdir("data"):
                if name.lower().endswith(".pdf"):
                    chosen = os.path.join("data", name)
                    break
    if not chosen:
        raise FileNotFoundError("No resume PDF found. Put it under ./data/ or set RESUME_PATH.")

    print(f"[startup] Building index from: {os.path.abspath(chosen)}")
    rag = RAGEngine.build_from_file(chosen)
    rag.save()
    print("[startup] Index built ✓")

@app.get("/health")
def health():
    info = {"status": "ok", "app": APP_NAME, "model": MODEL_NAME}
    try:
        rag = RAGEngine.load_or_none()
        info["rag_index"] = bool(rag)
    except Exception as e:
        info["rag_index_error"] = str(e)
    return info

def _log(event: Dict):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass

# special prompt intents
def normalize_intent(q: str) -> str:
    m = q.strip().lower()
    if m in {"skills", "skill", "my skills", "list skills"}:
        return ("List Pradeep Ponnam’s skills grouped as:\n"
                "• Programming Languages\n• Frameworks & Libraries\n• Databases\n"
                "• Cloud/DevOps\n• Tools & Platforms\n• Soft Skills\n"
                "Use short bullet points only.")
    if m in {"summary", "profile"}:
        return ("Give a crisp 4-bullet professional summary of Pradeep Ponnam "
                "(roles, domains, standout projects, tech stack).")
    return q

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    ip = client_ip(request)
    if not rate_limit_ok(ip):
        return ChatResponse(reply="Too many requests — please slow down.", session_id=req.session_id or "")

    sid = get_or_create_session(req.session_id)
    user_msg = (req.message or "").strip()
    if not user_msg:
        return ChatResponse(reply="Ask me anything about Pradeep Ponnam’s résumé.", session_id=sid, references=[])

    if user_msg.lower() == "clear history":
        SESSIONS[sid] = []
        return ChatResponse(reply="History cleared. What would you like to know about Pradeep’s profile?",
                            session_id=sid, references=[])

    try:
        rag = RAGEngine.load()
        q = normalize_intent(user_msg)
        answer, refs = rag.answer(q, model_name=MODEL_NAME, api_key=GROQ_API_KEY)
    except Exception as e:
        return ChatResponse(
            reply=f"Server error while answering (check FAISS/embeddings/Groq): {e}",
            session_id=sid, references=[]
        )

    SESSIONS[sid].append({"role": "user", "content": user_msg})
    SESSIONS[sid].append({"role": "bot", "content": answer})

    _log({"ts": int(time.time()), "ip": ip, "session": sid,
          "q": user_msg, "answer_len": len(answer), "refs": refs})

    return ChatResponse(reply=answer, session_id=sid, references=refs)

# websocket (optional)
class WSMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            msg = WSMessage(**data)
            sid = get_or_create_session(msg.session_id)
            text = (msg.message or "").strip()
            if not text:
                await ws.send_json({"reply": "Ask about Pradeep’s résumé.", "session_id": sid})
                continue
            if text.lower() == "clear history":
                SESSIONS[sid] = []
                await ws.send_json({"reply": "History cleared.", "session_id": sid})
                continue
            try:
                rag = RAGEngine.load()
                q = normalize_intent(text)
                answer, refs = rag.answer(q, model_name=MODEL_NAME, api_key=GROQ_API_KEY)
            except Exception as e:
                await ws.send_json({"reply": f"Error: {e}", "session_id": sid, "references": []})
                continue
            SESSIONS[sid].append({"role": "user", "content": text})
            SESSIONS[sid].append({"role": "bot", "content": answer})
            await ws.send_json({"reply": answer, "session_id": sid, "references": refs})
    except WebSocketDisconnect:
        return
