# api_server.py
import os
import io
import sys
from typing import Optional, Literal

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ===== Path setup =====
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.insert(0, repo_root)
sys.path.insert(0, current_dir)

sys.path.append(os.path.join(current_dir, 'GPT_SoVITS'))


# ===== Inference functions =====
from GPT_SoVITS.inference_webui import (
    change_gpt_weights,
    change_sovits_weights,
    get_tts_wav,
)

# ===== Constant resources =====
GPT_MODEL_PATH = "/workspace/zundamon-speech-api/zundamon_sovits/GPT_weights_v2/zudamon_style_1-e15.ckpt"
SOVITS_MODEL_PATH = "/workspace/zundamon-speech-api/zundamon_sovits/SoVITS_weights_v2/zudamon_style_1_e8_s96.pth"


REF_AUDIO_PATH = os.path.join(repo_root, "zundamon_sovits/reference", "reference.wav")
REF_TEXT = "流し切りが完全に入ればデバフの効果が付与される"
REF_LANGUAGE = "Japanese"

# ===== App =====
app = FastAPI(title="Zundamon TTS API (fixed ref)", version="1.0.0")

from fastapi.staticfiles import StaticFiles

app.mount("/reference", StaticFiles(directory=os.path.join(repo_root, "zundamon_sovits/reference")), name="reference")

_loaded = {"gpt": None, "sovits": None}

def _load_models():
    if not os.path.exists(GPT_MODEL_PATH):
        raise FileNotFoundError(f"Missing GPT model: {GPT_MODEL_PATH}")
    if not os.path.exists(SOVITS_MODEL_PATH):
        raise FileNotFoundError(f"Missing SoVITS model: {SOVITS_MODEL_PATH}")
    if not os.path.exists(REF_AUDIO_PATH):
        raise FileNotFoundError(f"Missing reference audio: {REF_AUDIO_PATH}")

    if _loaded["gpt"] != GPT_MODEL_PATH:
        change_gpt_weights(gpt_path=GPT_MODEL_PATH)
        _loaded["gpt"] = GPT_MODEL_PATH
    if _loaded["sovits"] != SOVITS_MODEL_PATH:
        change_sovits_weights(sovits_path=SOVITS_MODEL_PATH)
        _loaded["sovits"] = SOVITS_MODEL_PATH

@app.on_event("startup")
def _startup():
    _load_models()

class Health(BaseModel):
    status: str
    gpt: Optional[str]
    sovits: Optional[str]

@app.get("/health", response_model=Health)
def health():
    return {"status": "ok", "gpt": _loaded["gpt"], "sovits": _loaded["sovits"]}

AllowedLang = Literal["Korean", "English"]

@app.post("/synthesize")
def synthesize(
    target_text: str = Form(..., description="생성할 텍스트 (Korean/English)"),
    target_language: AllowedLang = Form(..., description="출력 언어: Korean | English"),
    top_p: float = Form(1.0),
    temperature: float = Form(1.0),
):
    try:
        if not target_text.strip():
            raise HTTPException(status_code=400, detail="target_text is empty")

        _load_models()

        synthesis = get_tts_wav(
            ref_wav_path=REF_AUDIO_PATH,
            prompt_text=REF_TEXT,
            prompt_language=REF_LANGUAGE,
            text=target_text,
            text_language=target_language,
            top_p=top_p,
            temperature=temperature,
        )

        last_sr, last_audio = None, None
        for sr, audio in synthesis:
            last_sr, last_audio = sr, audio

        if last_sr is None or last_audio is None:
            raise HTTPException(status_code=500, detail="No audio generated")

        import soundfile as sf
        out_buf = io.BytesIO()
        sf.write(out_buf, last_audio, last_sr, format="WAV")
        out_buf.seek(0)

        headers = {"Content-Disposition": 'inline; filename="output.wav"'}
        return StreamingResponse(out_buf, media_type="audio/wav", headers=headers)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===== uvicorn entrypoint =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )