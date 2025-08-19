import os
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

import faiss
import numpy as np
from pypdf import PdfReader
try:
    import docx
except Exception:
    docx = None

from sentence_transformers import SentenceTransformer
import httpx  # REST call to Groq

# Resolve project root as the folder one level above /app/
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
INDEX_PATH = DATA_DIR / "index.faiss"
META_PATH = DATA_DIR / "chunks.json"

_SYS_PROMPT = """You are Pradeep Ponnam’s résumé assistant.
Answer ONLY using the provided résumé context. If info is missing, say you do not have it.
Format:
- Use concise bullet points.
- End with a short "References:" list citing 1–3 brief snippets.
Never speculate or invent details.
"""

def ensure_data_dir() -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return str(DATA_DIR)

# ---------- Text extraction ----------

def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        text.append(txt)
    return "\n".join(text)

def _read_docx(path: str) -> str:
    if docx is None:
        raise RuntimeError("python-docx not installed.")
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)

def extract_text(path: str) -> str:
    p = path.lower()
    if p.endswith(".pdf"):
        return _read_pdf(path)
    if p.endswith(".docx"):
        return _read_docx(path)
    raise ValueError("Unsupported file type: " + path)

def clean_text(t: str) -> str:
    t = t.replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def chunk_text(text: str, max_words: int = 180, overlap: int = 40) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        j = min(len(words), i + max_words)
        chunk = " ".join(words[i:j])
        chunks.append(chunk)
        if j == len(words):
            break
        i = max(0, j - overlap)
    return [c for c in chunks if c.strip()]

# ---------- Embeddings (cached) ----------

_embedder_instance: SentenceTransformer | None = None

def _embedder() -> SentenceTransformer:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder_instance

def warm_embedder() -> None:
    _ = _embedder()  # preload weights at startup

def _embed(texts: List[str]) -> np.ndarray:
    model = _embedder()
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.array(vecs).astype("float32")

def _build_faiss(vectors: np.ndarray) -> faiss.IndexFlatIP:
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine (since normalized)
    index.add(vectors)
    return index

# ---------- RAG Engine ----------

class RAGEngine:
    def __init__(self, index, chunks: List[str]):
        self.index = index
        self.text_chunks = chunks

    @classmethod
    def build_from_file(cls, path: str) -> "RAGEngine":
        text = extract_text(path)
        text = clean_text(text)
        chunks = chunk_text(text)
        vectors = _embed(chunks)
        index = _build_faiss(vectors)
        return cls(index=index, chunks=chunks)

    def save(self):
        ensure_data_dir()
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump({"chunks": self.text_chunks}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls) -> "RAGEngine":
        if not (INDEX_PATH.exists() and META_PATH.exists()):
            raise FileNotFoundError("No index found for Pradeep’s résumé.")
        index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(index=index, chunks=meta["chunks"])

    @classmethod
    def load_or_none(cls):
        try:
            return cls.load()
        except Exception:
            return None

    def retrieve(self, query: str, k: int = 6) -> List[Tuple[int, float]]:
        qvec = _embed([query])
        D, I = self.index.search(qvec, k)
        return [(int(i), float(s)) for i, s in zip(I[0].tolist(), D[0].tolist()) if i != -1]

    # ---- Groq REST call ----
    def _groq_chat(self, *, prompt: str, model_name: str, api_key: str, temperature: float = 0.2) -> str:
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY")
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": _SYS_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 600,
            "stream": False,
        }
        with httpx.Client(timeout=60.0) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"]

    def answer(self, question: str, *, model_name: str, api_key: str, k: int = 6) -> Tuple[str, List[Dict[str, Any]]]:
        pairs = self.retrieve(question, k=k)
        selected = [self.text_chunks[i] for i, _ in pairs]
        context = "\n\n".join(f"- {c}" for c in selected) if selected else "- (no relevant snippets found)"
        refs = [{"chunk": i, "score": s, "preview": self.text_chunks[i][:160]} for i, s in pairs]

        prompt = f"""{_SYS_PROMPT}

[RESUME CONTEXT]
{context}

[USER QUESTION]
{question}

[RESPONSE INSTRUCTIONS]
- If the answer isn't in context, say: "I don't have that in the resume."
- Otherwise, answer directly and include a brief "References" list with short quoted snippets.
"""
        content = self._groq_chat(prompt=prompt, model_name=model_name, api_key=api_key, temperature=0.2)
        return content, refs
