import torch
import torchaudio

from .model_manager import ModelManager
from .text_analyzer import TextAnalyzer
from .audio_engine import AudioEngine

from config import (
    REF_AUDIO_PATH, REF_TEXT, REF_LANGUAGE,
    FALLBACK_LANG2CODE, device, is_half
)

from .audio_utils import get_spepc

class ZundamonTTS:
    def __init__(self):
        # 1. 전문가들 고용 (객체 생성)
        self.models = ModelManager()
        self.text_analyzer = TextAnalyzer()
        self.audio_engine = AudioEngine(self.models)
        
        # 2. 레퍼런스(목소리 샘플) 저장소
        self.ref_cache = {
            "prompt": None,
            "phones1": None,
            "bert1": None,
            "refers": None
        }

    # 🌟 [에러 해결 포인트!] lifespan에서 부르는 이름과 똑같이 맞췄다몬!
    def load_all_models(self):
        """사령관이 무기 관리자에게 모든 모델을 장전하라고 지시한다のだ!"""
        self.models.load_all_models()

    def lang(self, label_or_code: str) -> str:
        """언어 이름을 코드로 바꿔주는 유틸리티"""
        # 이제 dict_language는 모델 관리자(self.models)가 들고 있다몬!
        return self.models.dict_language.get(
            label_or_code, 
            FALLBACK_LANG2CODE.get(label_or_code, label_or_code)
        )

    def prepare_reference_cache(self):
        """레퍼런스 음성을 분석해서 캐시에 저장하는 복합 공정 지시!"""
        print("🟢 [사령관] 레퍼런스 오디오 캐시를 굽기 시작한다몬! 🍳")
        
        # (1) 번역가에게 레퍼런스 텍스트 분석 지시
        ref_lang_code = self.lang(REF_LANGUAGE)
        phones1, bert1 = self.text_analyzer.process(
            REF_TEXT, ref_lang_code, self.models.version
        )
        
        # (2) 오디오 파일 읽기 및 GPU 전송
        wav, sr = torchaudio.load(REF_AUDIO_PATH)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0).contiguous().to(device)
        if is_half: wav = wav.half()

        # (3) 모델을 이용해 특징 추출 (무기 관리자의 모델들을 빌려옴!)
        with torch.inference_mode():
            ssl_out = self.models.ssl_model.model(wav.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
            codes = self.models.vq_model.extract_latent(ssl_out)
            prompt = codes[0, 0].unsqueeze(0).to(device)
            if is_half: prompt = prompt.half()

        ref_spec = get_spepc(self.models.hps, REF_AUDIO_PATH).to(device)
        if is_half: ref_spec = ref_spec.half()

        # (4) 결과물을 사령관의 주머니(캐시)에 보관
        self.ref_cache["prompt"] = prompt
        self.ref_cache["phones1"] = phones1
        self.ref_cache["bert1"] = bert1
        self.ref_cache["refers"] = [ref_spec]
        
        print("✅ 레퍼런스 캐시 굽기 완료! 이제 즉시 합성 가능하다のだ!")

    def synthesize_with_cached_ref(self, target_text: str, target_language_label: str, top_p: float, temperature: float, top_k: int = 15, speed: float = 1.0):
        """실제 합성 요청이 왔을 때 전문가들을 총동원한다몬!"""
        
        # 캐시가 비어있으면 구워온다のだ!
        if any(self.ref_cache[k] is None for k in self.ref_cache):
            self.prepare_reference_cache()

        # 1. 번역가에게 타겟 텍스트 분석 지시
        lang_code = self.lang(target_language_label)
        phones2, bert2 = self.text_analyzer.process(
            target_text, lang_code, self.models.version
        )

        # 2. 공장장에게 최종 오디오 생성 지시
        sr, wav_buf = self.audio_engine.generate(
            ref_cache=self.ref_cache,
            target_phones=phones2,
            target_bert=bert2,
            speed=speed, top_k=top_k, top_p=top_p, temperature=temperature
        )
        
        return sr, wav_buf
    
zundamon_tts = ZundamonTTS()