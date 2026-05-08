import os

SOVITS_MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

# ===== model paths =====
GPT_MODEL_PATH = os.path.join(SOVITS_MODULE_PATH, "GPT_weights_v2", "zudamon_style_1-e15.ckpt")
SOVITS_MODEL_PATH = os.path.join(SOVITS_MODULE_PATH, "SoVITS_weights_v2", "zudamon_style_1_e8_s96.pth")

# ===== references =====
REF_AUDIO_PATH = os.path.join(SOVITS_MODULE_PATH, "reference", "reference.wav")
REF_TEXT = "、流し切りが完全に入れば、デバフの効果が付与される。"
REF_LANGUAGE = "Japanese"

FALLBACK_LANG2CODE = {
    "Japanese": "all_ja",
    "Korean": "all_ko",
    "English": "en",
    "Chinese": "all_zh",
}