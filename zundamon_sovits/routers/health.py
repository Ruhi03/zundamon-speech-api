from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from core.synthesize_core import zundamon_tts

router = APIRouter(tags=["System"])

class Health(BaseModel):
    status: str
    gpt: Optional[str]
    sovits: Optional[str]

@router.get("/health", response_model=Health)
def health():
    gpt_path = zundamon_tts.models.gpt_config.get("path") if zundamon_tts.models.gpt_config else None
    sovits_path = "v" + zundamon_tts.models.version if zundamon_tts.models.version else None
    return {"status": "ok", "gpt": gpt_path, "sovits": sovits_path}