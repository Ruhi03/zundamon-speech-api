BASE_PATH="/api/zundamon_sovits"

# 1. pretrained_models 체크 (.gitignore 말고 아무것도 없으면 다운로드)
TARGET_DIR="$BASE_PATH/GPT_SoVITS/pretrained_models"
mkdir -p "$TARGET_DIR"

FILE_COUNT=$(find "$TARGET_DIR" -maxdepth 1 ! -name ".gitignore" ! -path "$TARGET_DIR" | wc -l)

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "1. 사전 학습 모델이 비어있다몬! (또는 .gitignore뿐임) 다운로드 시작! 📥"
    git clone https://huggingface.co/lj1995/GPT-SoVITS temp_gptso
    cp -r temp_gptso/* "$TARGET_DIR/"
    rm -rf temp_gptso
else
    echo "1. 사전 학습 모델이 이미 존재한다몬! 스킵! ✅"
fi

# 2. G2PW 모델 체크 (G2PWModel 폴더가 없으면 다운로드)
G2PW_DIR="$BASE_PATH/GPT_SoVITS/text/G2PWModel"
if [ ! -d "$G2PW_DIR" ]; then
    echo "2. G2PW 모델 폴더가 없다몬! 다운로드 시작! 📥"
    wget https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/G2PWModel_1.1.zip -O G2PWModel_1.1.zip
    unzip G2PWModel_1.1.zip
    mv G2PWModel_1.1 G2PWModel
    mkdir -p $(dirname "$G2PW_DIR")
    mv G2PWModel "$G2PW_DIR"
    rm G2PWModel_1.1.zip
else
    echo "2. G2PW 모델이 이미 존재한다몬! 스킵! ✅"
fi

# 3. 즌다몬 가중치 폴더 체크 (GPT_weights_v2 또는 SoVITS_weights_v2 가 없으면 다운로드)
if [ ! -d "$BASE_PATH/GPT_weights_v2" ] || [ ! -d "$BASE_PATH/SoVITS_weights_v2" ]; then
    echo "3. 즈다몬 전용 가중치 폴더가 없다몬! 새로 가져온다몬! 📥"
    git clone https://huggingface.co/zunzunpj/zundamon_GPT-SoVITS temp_zunda
    cp -r temp_zunda/* "$BASE_PATH/"
    rm -rf temp_zunda
else
    echo "3. 즈다몬 가중치 폴더가 이미 완벽하게 세팅되어 있다몬! 스킵! ✅"
fi

echo "모든 검사가 끝났다몬! 서버를 시작하자のだ! 🚀"