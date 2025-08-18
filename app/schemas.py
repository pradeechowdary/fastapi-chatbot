from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # optional; server will assign if not provided

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    references: List[Dict[str, Any]] = []

class UploadResponse(BaseModel):
    message: str
    chunks: int
