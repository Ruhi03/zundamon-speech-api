import torch
from tools.my_utils import load_audio
from module.mel_processing import spectrogram_torch

def get_spepc(hps, filename):
    """오디오 파일을 읽어서 스펙트로그램(주파수 텐서)으로 변환하는 순수 유틸리티 함수다몬!"""
    
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    
    # 볼륨 정규화 (소리 찢어짐 방지)
    maxx = audio.abs().max()
    if (maxx > 1): 
        audio /= min(2, maxx)
        
    audio_norm = audio.unsqueeze(0)
    
    # 멜 스펙트로그램(Mel-spectrogram) 추출!
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    
    return spec