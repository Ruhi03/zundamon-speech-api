# 🟢 Zundamon Speech API

이 프로젝트는 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 기술을 기반으로 하며, 즌다몬의 목소리에 최적화되도록 파인튜닝 되었습니다.

## 🔌 API 명세서 (API Endpoints)

기본적으로 디스코드 봇과 소통하도록 만들어졌습니다.

### 음성 합성 요청 (POST /synthesize)

텍스트를 전송하면 즌다몬의 목소리로 합성된 오디오 스트림을 반환합니다.

- **요청 (Request)**: `application/json`

    ```json
    {
      "target_text": "음성으로 변환할 텍스트를 입력하세요.",
      "target_language": "Korean",  // 기본값: "Korean"
        (영어, 일본어 등 지원)
      "top_p": 0.7,                 // 기본값: 0.7
      "temperature": 0.8            // 기본값: 0.8
    }
    ```

- **응답 (Response)**: `audio/wav`

        생성된 음성 데이터가 StreamingResponse를 통해 즉시 스트리밍 됩니다.

## 🛠️ 설치 및 실행 방법
### 1단계: 사전 준비

    Docker & Docker Compose (Windows의 경우 Docker Desktop)

    NVIDIA GPU (빠른 추론을 위해 8GB 이상의 VRAM 권장)

    NVIDIA Container Toolkit (도커에서 GPU를 사용하기 위해 필수)

### 2-1단계 Download GPT-SoVITS Pretrained Models
```
cd /workspace

git clone https://huggingface.co/lj1995/GPT-SoVITS temp_gptso

cp -r temp_gptso/* /workspace/zundamon-speech-api/zundamon_sovits/GPT_SoVITS/pretrained_models/

rm -rf temp_gptso
```

### 2-2단계 Download G2PW Models
```
cd /workspace

wget https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/G2PWModel_1.1.zip -O G2PWModel_1.1.zip

unzip G2PWModel_1.1.zip

mv G2PWModel_1.1 G2PWModel

mv G2PWModel /workspace/zundamon-speech-api/zundamon_sovits/GPT_SoVITS/text/

rm G2PWModel_1.1.zip
```

### 2-3단계 Download Zundamon Fine-Tuned Model
```
cd /workspace

git clone https://huggingface.co/zunzunpj/zundamon_GPT-SoVITS temp_zunda

cp -r temp_zunda/* /workspace/zundamon-speech-api/zundamon_sovits/

rm -rf temp_zunda
```

### 3단계: 서버 실행 (Docker Compose)

모든 준비가 완료되었다면, 다음 명령어를 통해 API 서버가 실행합니다

```Bash
docker-compose up -d
```
## 📜 라이선스 정보

이 소프트웨어는 다음과 같은 오픈소스 소프트웨어를 포함하고 있습니다:

- GPT-SoVITS (MIT License)
- GPT-SoVITS Pretrained Models (MIT License)
- G2PW Model (Apache 2.0 License)
- UVR5 (Voice Cleaning) (MIT License)
- Faster Whisper Large V3 (MIT License)
- FastAPI (MIT License)

해당 소프트웨어들은 각각의 라이선스 조항에 따라 제공됩니다.

즌다몬(Zundamon) 음성 모델에 대한 라이선스는 다음 규약을 따릅니다:
https://zunko.jp/con_ongen_kiyaku.html