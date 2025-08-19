import os
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from .schemas import ChatRequest, ChatResponse
from .rag_utils import RAGEngine, ensure_data_dir, warm_embedder

load_dotenv()

APP_NAME   = os.getenv("APP_NAME", "Pradeep • Résumé Bot")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
RESUME_PATH = os.getenv("RESUME_PATH", str(Path(__file__).resolve().parents[1] / "data" / "PonnamcCV.pdf"))

app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# static + index
ROOT_DIR = Path(__file__).resolve().parents[1]
app.mount("/static", StaticFiles(directory=str(ROOT_DIR / "static")), name="static")

@app.get("/")
def root():
    return FileResponse(str(ROOT_DIR / "templates" / "index.html"))

@app.head("/")  # for Render health checks if they ping HEAD /
def root_head():
    return Response(status_code=200)

# sessions
History = List[Dict[str, str]]
SESSIONS: Dict[str, History] = {}

def get_or_create_session(session_id: Optional[str]) -> str:
    if session_id and session_id in SESSIONS:
        return session_id
    sid = str(uuid.uuid4())
    SESSIONS[sid] = []
    return sid

# startup: build index if missing, warm the embedder
@app.on_event("startup")
def _startup():
    ensure_data_dir()
    try:
        RAGEngine.load()  # already built?
    except Exception:
        chosen = None
        env_path = RESUME_PATH.strip()
        if env_path and Path(env_path).exists():
            chosen = env_path
        else:
            # try default under /app/data
            default = ROOT_DIR / "data" / "PonnamcCV.pdf"
            if default.exists():
                chosen = str(default)
            else:
                # pick first pdf under data/
                pdfs = list((ROOT_DIR / "data").glob("*.pdf"))
                if pdfs:
                    chosen = str(pdfs[0])
        if not chosen:
            raise FileNotFoundError("No resume PDF found. Put it in ./data or set RESUME_PATH.")
        print(f"[startup] Building index from: {chosen}")
        rag = RAGEngine.build_from_file(chosen)
        rag.save()
        print("[startup] Index built ✓")

    warm_embedder()
    print("[startup] Embedder warmed ✓")

@app.get("/health")
def health():
    info = {"status": "ok", "app": APP_NAME, "model": MODEL_NAME}
    try:
        rag = RAGEngine.load_or_none()
        info["rag_index"] = bool(rag)
    except Exception as e:
        info["rag_index_error"] = str(e)
    return info

# chat
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    sid = get_or_create_session(req.session_id)
    user_msg = (req.message or "").strip()
    if not user_msg:
        return ChatResponse(reply="Ask me anything about Pradeep Ponnam’s résumé.", session_id=sid, references=[])

    if user_msg.lower() == "clear history":
        SESSIONS[sid] = []
        return ChatResponse(reply="History cleared. What would you like to know?", session_id=sid, references=[])

    try:
        rag = RAGEngine.load()
        answer, refs = rag.answer(user_msg, model_name=MODEL_NAME, api_key=GROQ_API_KEY)
    except Exception as e:
        return ChatResponse(
            reply=f"Server error while answering (check FAISS/embeddings/Groq): {e}",
            session_id=sid, references=[]
        )

    SESSIONS[sid].append({"role": "user", "content": user_msg})
    SESSIONS[sid].append({"role": "bot", "content": answer})
    return ChatResponse(reply=answer, session_id=sid, references=refs)

# optional websocket
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
                answer, refs = rag.answer(text, model_name=MODEL_NAME, api_key=GROQ_API_KEY)
            except Exception as e:
                await ws.send_json({"reply": f"Error: {e}", "session_id": sid, "references": []})
                continue
            SESSIONS[sid].append({"role": "user", "content": text})
            SESSIONS[sid].append({"role": "bot", "content": answer})
            await ws.send_json({"reply": answer, "session_id": sid, "references": refs})
    except WebSocketDisconnect:
        return
