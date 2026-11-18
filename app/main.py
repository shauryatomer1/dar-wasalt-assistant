import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .llm import build_messages, stream_answer
from .rag import Retriever

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="DarGlobal and Wasalt Assistant")
R = Retriever(use_dense=os.getenv("DENSE_RETRIEVAL", "1") == "1")  # set 0 on tiny hosts to save RAM
HITS = defaultdict(deque)
RATE = int(os.getenv("RATE_LIMIT_PER_MIN", "12"))


class Msg(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=600)
    history: list[Msg] = []


def sse(obj):
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def fallback_answer(chunks):
    if not chunks:
        return "I couldn't find anything relevant in the collected DarGlobal / Wasalt data."
    return ("The language model is temporarily unavailable, so here are the most relevant passages I found:\n\n"
            + "\n\n".join(f"**[{i}] {c['title']}** - {c['text'][:350].strip()}..." for i, c in enumerate(chunks[:3], 1)))


@app.get("/api/health")
def health():
    return {"status": "ok", "chunks": len(R.chunks), "listings": len(R.listings), "dense": R.emb is not None,
            "llm_configured": bool(os.getenv("OPENROUTER_API_KEY"))}


@app.get("/api/stats")
def stats():
    by = defaultdict(int)
    for c in R.chunks:
        by[f"{c['source']}:{c['kind']}"] += 1
    return {"chunks": by, "listings": len(R.listings)}


@app.post("/api/chat")
async def chat(body: ChatIn, request: Request):
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
    now = time.time()
    q = HITS[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE:
        return JSONResponse({"error": "Too many requests - please wait a moment."}, status_code=429)
    q.append(now)

    last_user = next((m.content for m in reversed(body.history) if m.role == "user"), None)
    res = R.retrieve(body.message, history_q=last_user)
    chunks = res["chunks"]
    history = [m.model_dump() for m in body.history]

    async def gen():
        yield sse({"type": "sources", "sources": [
            {"n": i, "title": c["title"], "url": c["url"], "source": c["source"], "kind": c["kind"]}
            for i, c in enumerate(chunks, 1)]})
        try:
            async for kind, val in stream_answer(build_messages(body.message, chunks, history)):
                yield sse({"type": "token", "text": val} if kind == "token" else {"type": "model", "model": val})
        except RuntimeError as e:
            yield sse({"type": "token", "text": fallback_answer(chunks)})
            yield sse({"type": "notice", "text": str(e)[:200]})
        yield sse({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
