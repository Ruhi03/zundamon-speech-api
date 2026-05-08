import os
import io
import wave
import torch
import torchaudio
from typing import Any, Dict

from config import (
    GPT_MODEL_PATH, SOVITS_MODEL_PATH,
    REF_AUDIO_PATH, REF_TEXT, REF_LANGUAGE,
    FALLBACK_LANG2CODE, is_half, device, ssl_model
)

from GPT_SoVITS.inference_webui import (
    change_gpt_weights, change_sovits_weights,
    vq_model, hps, version, dict_language,
    hz, max_sec, t2s_model,
)

from text_processor import get_phones_and_bert
from audio_utils import get_spepc

class ZundamonTTS:
    def __init__(self):
        self.loaded: Dict[str, Any] = {"gpt": None, "sovits": None}
        self.ref_cache: Dict[str, Any] = {
            "prompt": None, 
            "phones1": None, 
            "bert1": None, 
            "refers": None
        }

    def lang(self, label_or_code: str) -> str:
        return dict_language.get(label_or_code, FALLBACK_LANG2CODE.get(label_or_code, label_or_code))

    def prepare_reference_cache(self):
        ref_lang_code = self.lang(REF_LANGUAGE)
        phones1, bert1, _ = get_phones_and_bert(REF_TEXT, ref_lang_code, version)
        
        if hasattr(bert1, "to"):
            bert1 = bert1.to(device) # type: ignore
            
            if is_half:
                bert1 = bert1.half() # type: ignore

        wav, sr = torchaudio.load(REF_AUDIO_PATH) # type: ignore

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000) # type: ignore
        
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        wav = wav.squeeze(0).contiguous().to(device) # type: ignore

        if is_half:
            wav = wav.half() # type: ignore

        with torch.inference_mode():
            ssl_out = ssl_model.model(wav.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
            codes = vq_model.extract_latent(ssl_out)
            prompt = codes[0, 0].unsqueeze(0).to(device)
            
            if is_half:
                prompt = prompt.half()

        ref_spec = get_spepc(hps, REF_AUDIO_PATH).to(device)
        
        if is_half:
            ref_spec = ref_spec.half() # type: ignore

        self.ref_cache["prompt"] = prompt
        self.ref_cache["phones1"] = phones1
        self.ref_cache["bert1"] = bert1
        self.ref_cache["refers"] = [ref_spec]

    def load_models(self):
        if not os.path.exists(GPT_MODEL_PATH):
            raise FileNotFoundError(f"Missing GPT model: {GPT_MODEL_PATH}")
        
        if not os.path.exists(SOVITS_MODEL_PATH):
            raise FileNotFoundError(f"Missing SoVITS model: {SOVITS_MODEL_PATH}")
        
        if not os.path.exists(REF_AUDIO_PATH):
            raise FileNotFoundError(f"Missing reference audio: {REF_AUDIO_PATH}")

        if self.loaded["gpt"] != GPT_MODEL_PATH:
            change_gpt_weights(gpt_path=GPT_MODEL_PATH)
            self.loaded["gpt"] = GPT_MODEL_PATH
        
        if self.loaded["sovits"] != SOVITS_MODEL_PATH:
            change_sovits_weights(sovits_path=SOVITS_MODEL_PATH)
            self.loaded["sovits"] = SOVITS_MODEL_PATH

        self.prepare_reference_cache()

    def dynamic_early_stop(self, phones_len: int) -> int:
        est_secs = max(1, int(phones_len / 12))
        return min(hz * max_sec, int(hz * est_secs * 2.4))

    def synthesize_with_cached_ref(self, target_text: str, target_language_label: str, top_p: float, temperature: float, top_k: int = 15, speed: float = 1.0) -> tuple[int, io.BytesIO]:
        if any(self.ref_cache[k] is None for k in ("prompt", "phones1", "bert1", "refers")):
            self.prepare_reference_cache()

        prompt = self.ref_cache["prompt"]
        phones1 = self.ref_cache["phones1"]
        bert1 = self.ref_cache["bert1"]
        refers = self.ref_cache["refers"]

        lang_code = self.lang(target_language_label)
        phones2, bert2, _ = get_phones_and_bert(target_text, lang_code, version)
        
        if hasattr(bert2, "to"):
            bert2 = bert2.to(device) # type: ignore
            
            if is_half:
                bert2 = bert2.half() # type: ignore

        bert = (torch.cat([bert1, bert2], dim=1)).unsqueeze(0) # type: ignore
        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0) # type: ignore
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        
        early_stop_num = self.dynamic_early_stop(len(phones2))
        
        with torch.inference_mode():
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids, all_phoneme_len, prompt, bert, # type: ignore
                top_k=top_k, top_p=top_p, temperature=temperature, early_stop_num=early_stop_num, # type: ignore
            )
            
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)

            audio = (
                vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refers, speed=speed) # type: ignore
                .detach().cpu().numpy()[0, 0]
            )

        maxabs = float(abs(audio).max())
        
        if maxabs > 1.0:
            audio = audio / maxabs
        
        int16_pcm = (audio * 32768.0).astype('int16')
        
        buf = io.BytesIO()
        
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(hps.data.sampling_rate)
            wf.writeframes(int16_pcm.tobytes())
        buf.seek(0)
        
        return hps.data.sampling_rate, buf