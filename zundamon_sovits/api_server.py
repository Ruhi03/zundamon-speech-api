# api_server.py
import os
import io
import sys
from typing import Optional, Literal

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import traceback

# ===== Path setup =====
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.insert(0, repo_root)
sys.path.insert(0, current_dir)
sys.path.append(os.path.join(current_dir, 'GPT_SoVITS'))

# ===== Inference functions & globals from GPT-SoVITS =====
from GPT_SoVITS.inference_webui import (
    change_gpt_weights,
    change_sovits_weights,
    get_tts_wav,          # (안 써도 무방하지만 남겨둠)
    get_phones_and_bert,
    get_spepc,
    ssl_model,
    vq_model,
    hps,
    device,
    is_half,
    t2s_model,
    hz,
    max_sec,
    version,
    dict_language,
)

# ===== Constant resources =====
GPT_MODEL_PATH = "/workspace/zundamon-speech-api/zundamon_sovits/GPT_weights_v2/zudamon_style_1-e15.ckpt"
SOVITS_MODEL_PATH = "/workspace/zundamon-speech-api/zundamon_sovits/SoVITS_weights_v2/zudamon_style_1_e8_s96.pth"
REF_AUDIO_PATH = os.path.join(repo_root, "zundamon_sovits/reference", "reference.wav")
REF_TEXT = "流し切りが完全に入ればデバフの効果が付与される"
REF_LANGUAGE = "Japanese"  # 라벨이든 내부코드든 OK(아래 _lang()이 처리)

# ===== App =====
app = FastAPI(title="Zundamon TTS API (fixed ref)", version="1.0.0")

# ===== Language mapping helper =====
FALLBACK_LANG2CODE = {
    "Japanese": "all_ja",
    "Korean": "all_ko",
    "English": "en",
    "Chinese": "all_zh",
}
def _lang(label_or_code: str) -> str:
    """
    WebUI 라벨('Japanese','Korean','English' ...) → 내부 코드('all_ja','all_ko','en' ...)
    dict_language(v1/v2)에 먼저 질의 후, 폴백 테이블 참조. 없으면 원문 반환.
    """
    return dict_language.get(label_or_code, FALLBACK_LANG2CODE.get(label_or_code, label_or_code))

# ===== Global state =====
_loaded = {"gpt": None, "sovits": None}

# 참조(레퍼런스) 캐시: 부팅/모델 교체 시 1회 준비
_REF_CACHE = {
    "prompt": None,   # (1, semantic_len) Tensor[device]
    "phones1": None,  # list[int]
    "bert1": None,    # Tensor[device]
    "refers": None,   # List[Tensor] (decode용 spec)
}

def _prepare_reference_cache():
    """레퍼런스 음성/텍스트를 한 번만 전처리해 GPU 캐시에 보관."""
    # 1) 프롬프트 텍스트 전처리
    ref_lang_code = _lang(REF_LANGUAGE)
    phones1, bert1, _ = get_phones_and_bert(REF_TEXT, ref_lang_code, version)
    if hasattr(bert1, "to"):
        bert1 = bert1.to(device)
        if is_half:
            bert1 = bert1.half()

    # 2) 레퍼런스 음성 → HuBERT → VQ (prompt 코드)
    import torchaudio, torch
    wav, sr = torchaudio.load(REF_AUDIO_PATH)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0).contiguous().to(device)
    if is_half:
        wav = wav.half()

    with torch.inference_mode():
        ssl_out = ssl_model.model(wav.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        codes = vq_model.extract_latent(ssl_out)
        prompt = codes[0, 0].unsqueeze(0).to(device)
        if is_half:
            prompt = prompt.half()

    # 3) 디코더 참조 스펙
    ref_spec = get_spepc(hps, REF_AUDIO_PATH).to(device)
    if is_half:
        ref_spec = ref_spec.half()

    _REF_CACHE["prompt"] = prompt
    _REF_CACHE["phones1"] = phones1
    _REF_CACHE["bert1"] = bert1
    _REF_CACHE["refers"] = [ref_spec]

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

    # 모델이 바뀌었을 수 있으니 참조 캐시도 재계산
    _prepare_reference_cache()

@app.on_event("startup")
def _startup():
    _load_models()

# ===== Inference helpers =====
def _dynamic_early_stop(phones_len: int) -> int:
    """문장 길이 기반으로 GPT 샘플링 상한 축소."""
    est_secs = max(1, int(phones_len / 12))         # 대략 phoneme 12 ≈ 1초
    return min(hz * max_sec, int(hz * est_secs * 2.4))

def _synthesize_with_cached_ref(
    target_text: str,
    target_language_label: str,
    top_p: float,
    temperature: float,
    top_k: int = 15,
    speed: float = 1.0,
):
    """
    참조 준비(SSL/VQ/텍스트)를 캐시로 재사용하여 빠르게 합성.
    반환: (sr, io.BytesIO[wav])
    """
    import wave
    import numpy as np
    import torch

    if any(_REF_CACHE[k] is None for k in ("prompt", "phones1", "bert1", "refers")):
        _prepare_reference_cache()

    prompt = _REF_CACHE["prompt"]
    phones1 = _REF_CACHE["phones1"]
    bert1 = _REF_CACHE["bert1"]
    refers = _REF_CACHE["refers"]

    # 입력 텍스트 전처리 (라벨→코드 매핑)
    lang_code = _lang(target_language_label)
    phones2, bert2, _ = get_phones_and_bert(target_text, lang_code, version)
    if hasattr(bert2, "to"):
        bert2 = bert2.to(device)
        if is_half:
            bert2 = bert2.half()

    # 결합
    bert = (torch.cat([bert1, bert2], dim=1)).unsqueeze(0)  # (1, 1024, T)
    all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
    all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

    # GPT 샘플링
    early_stop_num = _dynamic_early_stop(len(phones2))
    with torch.inference_mode():
        pred_semantic, idx = t2s_model.model.infer_panel(
            all_phoneme_ids,
            all_phoneme_len,
            prompt,
            bert,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            early_stop_num=early_stop_num,
        )
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)

        # 디코드
        audio = (
            vq_model.decode(
                pred_semantic,
                torch.LongTensor(phones2).to(device).unsqueeze(0),
                refers,
                speed=speed,
            )
            .detach()
            .cpu()
            .numpy()[0, 0]
        )

    # int16 변환
    maxabs = float(abs(audio).max())
    if maxabs > 1.0:
        audio = audio / maxabs
    int16_pcm = (audio * 32768.0).astype('int16')

    # WAV로 패킹
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(hps.data.sampling_rate)
        wf.writeframes(int16_pcm.tobytes())
    buf.seek(0)
    return hps.data.sampling_rate, buf

# ===== API =====
class Health(BaseModel):
    status: str
    gpt: Optional[str]
    sovits: Optional[str]

@app.get("/health", response_model=Health)
def health():
    return {"status": "ok", "gpt": _loaded["gpt"], "sovits": _loaded["sovits"]}

AllowedLang = Literal["Korean", "English"]  # 필요시 "Japanese" 추가 또는 str로 변경

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

        # 캐시 사용 합성
        sr, wav_buf = _synthesize_with_cached_ref(
            target_text=target_text.strip(),
            target_language_label=target_language,
            top_p=top_p,
            temperature=temperature,
            top_k=15,
            speed=1.0,
        )

        headers = {"Content-Disposition": 'inline; filename="output.wav"'}
        return StreamingResponse(wav_buf, media_type="audio/wav", headers=headers)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
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