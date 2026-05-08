from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

from synthesize import router as synthesize_router, zundamon_tts

app = FastAPI(title="Zundamon Speech API", version="1.0.0")

app.include_router(synthesize_router)

# ===== API =====
class Health(BaseModel):
    status: str
    gpt: Optional[str]
    sovits: Optional[str]

@app.get("/health", response_model=Health)
def health():
    return {"status": "ok", "gpt": zundamon_tts.loaded["gpt"], "sovits": zundamon_tts.loaded["sovits"]}