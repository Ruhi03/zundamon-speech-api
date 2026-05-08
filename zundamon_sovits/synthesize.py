from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import traceback

from synthesize_core import ZundamonTTS

router = APIRouter()
zundamon_tts = ZundamonTTS()

# 1) 디스코드 봇이 보낼 JSON 데이터 규격 만들기
class SynthesizeRequest(BaseModel):
    target_text: str
    target_language: str = "Korean" # 기본값 설정 가능
    top_p: float = 0.7
    temperature: float = 0.8

@router.post("/synthesize")
def synthesize(req: SynthesizeRequest) -> StreamingResponse:
    try:
        if not req.target_text.strip():
            raise HTTPException(status_code=400, detail="target_text is empty")

        # 캐시 사용 합성
        sr, wav_buf = zundamon_tts.synthesize_with_cached_ref(
            target_text=req.target_text.strip(),
            target_language_label=req.target_language,
            top_p=req.top_p,
            temperature=req.temperature,
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