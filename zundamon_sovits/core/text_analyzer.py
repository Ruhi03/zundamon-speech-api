import torch
import LangSegment
from text import cleaned_text_to_sequence
from text.cleaner import clean_text

# config에서 필요한 하드웨어 설정 가져오기
from config import device, is_half

class TextAnalyzer:
    def __init__(self):
        print("🟢 [TextAnalyzer] 텍스트 분석가(번역가)를 고용했다몬! 📝")
        # 기본적으로 한국어, 일본어, 영어만 감지하도록 필터 세팅
        LangSegment.setfilters(["ja", "en", "ko"])

    def _clean_text_inf(self, text, language, version):
        """텍스트를 발음 기호(Phones)로 변환하는 내부 함수다몬"""
        phones, word2ph, norm_text = clean_text(text, language, version)
        phones = cleaned_text_to_sequence(phones, version)
        return phones, word2ph, norm_text

    def _get_bert_inf(self, phones, word2ph, norm_text, language):
        """텍스트의 문맥(BERT 피처)을 추출하는 내부 함수다のだ"""
        language = language.replace("all_", "")
        
        # 중국어일 때만 실제 BERT를 쓰고, 한/일/영은 0으로 채운 빈 텐서를 반환해 (GPT-SoVITS 기본 구조)
        if language == "zh":
            from text.chinese_bert import get_bert_feature # (경로는 원본에 맞게!)
            bert = get_bert_feature(norm_text, word2ph).to(device)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half else torch.float32,
            ).to(device)
            
        return bert

    def process(self, text: str, language: str, version: str):
        """
        👑 ZundamonTTS 사령관이 직접 호출할 유일한 메인 함수다のだ!
        입력: 텍스트, 언어, 모델 버전(v1/v2)
        출력: 발음 기호(phones), 문맥 텐서(bert)
        """
        # 1. 단일 언어(한/일/영) 모드일 때의 초고속 처리
        if language in {"en", "all_ja", "all_ko"}:
            language = language.replace("all_", "")
            if language == "en":
                LangSegment.setfilters(["en"])
                formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
            else:
                formattext = text
                
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
                
            phones, word2ph, norm_text = self._clean_text_inf(formattext, language, version)
            bert = self._get_bert_inf(phones, word2ph, norm_text, language)
            
            return phones, bert
            
        # 2. 다국어(Auto) 짬뽕 모드일 때의 정밀 처리
        langlist, textlist = [], []
        LangSegment.setfilters(["ja", "en", "ko"])
        
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    langlist.append(language)
                textlist.append(tmp["text"])
                
        phones_list = []
        bert_list = []
        norm_text_list = []
        
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = self._clean_text_inf(textlist[i], lang, version)
            bert = self._get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
            
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        
        return phones, bert