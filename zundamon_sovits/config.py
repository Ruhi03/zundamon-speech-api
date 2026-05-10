import os
import torch

# ==========================================
# 1. 📂 시스템 경로 및 파일 설정 (Paths)
# ==========================================
SOVITS_MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

# 🧠 인공지능 뇌(모델) 경로
GPT_MODEL_PATH = os.path.join(SOVITS_MODULE_PATH, "GPT_weights_v2", "zudamon_style_1-e15.ckpt")
SOVITS_MODEL_PATH = os.path.join(SOVITS_MODULE_PATH, "SoVITS_weights_v2", "zudamon_style_1_e8_s96.pth")
cnhubert_base_path = os.environ.get("cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base")

# 🎙️ 기본 레퍼런스(목소리 샘플) 설정
REF_AUDIO_PATH = os.path.join(SOVITS_MODULE_PATH, "reference", "reference.wav")
REF_TEXT = "、流し切りが完全に入れば、デバフの効果が付与される。"
REF_LANGUAGE = "Japanese"

# ==========================================
# 2. ⚡ 하드웨어 및 연산 설정 (Device & Precision)
# ==========================================
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# GPU가 있으면 절반 정밀도(FP16)를 사용하여 속도 2배, 메모리 절반 최적화!
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
dtype = torch.float16 if is_half else torch.float32


# ==========================================
# 3. 🌐 언어 매핑 사전 (Language Dictionaries)
# ==========================================
# API에서 직관적으로 쓰기 위한 폴백 매핑
FALLBACK_LANG2CODE = {
    "Japanese": "all_ja",
    "Korean": "all_ko",
    "English": "en",
    "Chinese": "all_zh",
}

# (웹 UI 번역기 i18n 찌꺼기 완벽 제거!)
dict_language_v1 = {
    "Chinese": "all_zh",
    "English": "en",
    "Japanese": "all_ja",
    "Chinese+English": "zh",
    "Japanese+English": "ja",
    "Auto": "auto",
}

# v2 모델용 언어 코드 (즈다몬은 주로 이쪽을 쓴다のだ!)
dict_language_v2 = {
    "Chinese": "all_zh",
    "English": "en",
    "Japanese": "all_ja",
    "Cantonese": "all_yue",
    "Korean": "all_ko",
    "Chinese+English": "zh",
    "Japanese+English": "ja",
    "Cantonese+English": "yue",
    "Korean+English": "ko",
    "Auto": "auto",
}