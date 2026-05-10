from contextlib import asynccontextmanager
from fastapi import FastAPI

from core.synthesize_core import zundamon_tts

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🟢 서버 시작! 즈다몬 모델을 메모리에 장전한다몬... 🧠⚡")
    
    zundamon_tts.load_all_models()
    zundamon_tts.prepare_reference_cache() 
    
    print("✅ 모델 장전 완료! 언제든 말할 준비가 되었다のだ!")
    
    yield

    print("🔴 서버가 종료된다몬... 🧹")