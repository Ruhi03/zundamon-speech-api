import torch
import numpy as np
import io
import wave

from config import device

class AudioEngine:
    def __init__(self, model_manager):
        print("🟢 [AudioEngine] 오디오 생산 공장장 고용 완료몬! 🎧")
        self.models = model_manager

    def _dynamic_early_stop(self, phoneme_length):
        return int(phoneme_length * self.models.hz / 25) + int(self.models.hz * self.models.max_sec)

    def generate(self, ref_cache, target_phones, target_bert, speed=1.0, top_k=5, top_p=1.0, temperature=1.0):
        print("⚙️ 오디오 생성 공정 시작한다のだ...")
        
        prompt = ref_cache["prompt"]
        phones1 = ref_cache["phones1"]
        bert1 = ref_cache["bert1"]
        refers = ref_cache["refers"]

        bert2 = target_bert
        phones2 = target_phones

        bert = torch.cat([bert1, bert2], dim=1).unsqueeze(0)
        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        
        early_stop_num = self._dynamic_early_stop(len(phones2))

        with torch.inference_mode():
            pred_semantic, idx = self.models.t2s_model.model.infer_panel(
                all_phoneme_ids, all_phoneme_len, prompt, bert, 
                top_k=top_k, top_p=top_p, temperature=temperature, early_stop_num=early_stop_num,
            )
            
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)

            audio = (
                self.models.vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refers, speed=speed)
                .detach().cpu().numpy()[0, 0]
            )

        maxabs = float(abs(audio).max())
        if maxabs > 1.0:
            audio = audio / maxabs
            
        int16_pcm = (audio * 32768.0).astype('int16')
        
        # 🌟 여기서부터가 네가 짚어준 핵심 포인트다몬! 🌟
        buf = io.BytesIO()
        sampling_rate = self.models.hps.data.sampling_rate # 모델 관리자에게서 샘플링 레이트 가져오기!
        
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sampling_rate)
            wf.writeframes(int16_pcm.tobytes())
            
        buf.seek(0) # 스트림의 읽기 위치를 맨 앞으로 되돌려놔야 FastAPI가 처음부터 읽어간다のだ!
        
        print("✅ 오디오 생성 및 WAV 변환 완료のだ!")
        return sampling_rate, buf