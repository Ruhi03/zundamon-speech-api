import torch
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from module.models import SynthesizerTrn
from feature_extractor import cnhubert

# config에서 필요한 설정값들을 가져온다몬
from config import (
    GPT_MODEL_PATH, 
    SOVITS_MODEL_PATH,
    cnhubert_base_path,
    is_half, 
    device,
    dict_language_v1,
    dict_language_v2
)

# 딕셔너리를 객체 속성처럼 쓸 수 있게 해주는 유틸 (보통 utils나 별도 파일에 있음)
class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)
            
    def __getattr__(self, item):
        try: return self[item]
        except KeyError: raise AttributeError(f"Attribute {item} not found")
        
    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)


class ModelManager:
    def __init__(self):
        print("🟢 [ModelManager] 즈다몬의 뇌(모델) 관리자를 호출했다몬! 🧠")
        # --- 3대장 모델 저장소 ---
        self.t2s_model = None    # GPT 모델 (운율/흐름)
        self.vq_model = None     # SoVITS 모델 (음색/발음)
        self.ssl_model = None    # HuBERT 모델 (오디오 특징 추출)
        
        # --- 메타 데이터 ---
        self.hps = None
        self.gpt_config = None
        self.max_sec = None
        self.version = None
        self.hz = 50
        self.dict_language = None

    def load_all_models(self):
        """사령관(ZundamonTTS)이 '전원 장전!' 명령을 내릴 때 쓰는 함수다몬!"""
        print("🚀 모든 인공지능 모델 로딩을 시작한다のだ!")
        self._load_ssl_model()
        self._load_gpt_model(GPT_MODEL_PATH)
        self._load_sovits_model(SOVITS_MODEL_PATH)
        print("✅ 즈다몬의 두뇌 3종 세트가 완벽하게 장전되었다몬!")

    def _load_ssl_model(self):
        print(f"🟢 [SSL] 특징 추출 모델 로딩 중... 🎧")
        cnhubert.cnhubert_base_path = cnhubert_base_path
        self.ssl_model = cnhubert.get_model()
        
        if is_half:
            self.ssl_model = self.ssl_model.half().to(device)
        else:
            self.ssl_model = self.ssl_model.to(device)

    def _load_gpt_model(self, gpt_path: str):
        print(f"🟢 [GPT] 운율 예측 모델 로딩 중... 🗣️")
        dict_s1 = torch.load(gpt_path, map_location="cpu", weights_only=False)
        self.gpt_config = dict_s1["config"]
        self.max_sec = self.gpt_config["data"]["max_sec"]
        
        self.t2s_model = Text2SemanticLightningModule(self.gpt_config, "****", is_train=False)
        self.t2s_model.load_state_dict(dict_s1["weight"])
        
        if is_half:
            self.t2s_model = self.t2s_model.half()
        self.t2s_model = self.t2s_model.to(device)
        self.t2s_model.eval()

    def _load_sovits_model(self, sovits_path: str):
        print(f"🟢 [SoVITS] 음색 합성 모델 로딩 중... 🎙️")
        dict_s2 = torch.load(sovits_path, map_location="cpu", weights_only=False)
        self.hps = DictToAttrRecursive(dict_s2["config"])
        self.hps.model.semantic_frame_rate = "25hz"
        
        if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
            self.hps.model.version = "v1"
        else:
            self.hps.model.version = "v2"
        self.version = self.hps.model.version
        
        self.vq_model = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        )
        
        if ("pretrained" not in sovits_path):
            del self.vq_model.enc_q
            
        if is_half:
            self.vq_model = self.vq_model.half().to(device)
        else:
            self.vq_model = self.vq_model.to(device)
            
        self.vq_model.eval()
        self.vq_model.load_state_dict(dict_s2["weight"], strict=False)
        self.dict_language = dict_language_v1 if self.version == 'v1' else dict_language_v2