import os
import sys
import nltk

# 현재 이 파일(__init__.py)이 있는 폴더 (.../zundamon_sovits)
SOVITS_MODULE_PATH = os.path.dirname(os.path.abspath(__file__))

# 최상위 프로젝트 폴더 (.../zundamon-speech-api)
REPO_ROOT = os.path.dirname(SOVITS_MODULE_PATH)

# GPT_SoVITS 내부 코어 경로
GPT_SOVITS_CORE_PATH = os.path.join(SOVITS_MODULE_PATH, "GPT_SoVITS")

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SOVITS_MODULE_PATH)
sys.path.insert(0, GPT_SOVITS_CORE_PATH)
os.chdir(SOVITS_MODULE_PATH)

nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('cmudict', quiet=True)