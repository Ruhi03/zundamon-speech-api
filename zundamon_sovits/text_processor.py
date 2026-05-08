import torch
import LangSegment
from text import cleaned_text_to_sequence
from text.cleaner import clean_text

from config import device, is_half, dtype

def clean_text_inf(text, language, version):
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text

def get_bert_inf(phones, word2ph, norm_text, language):
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert

def get_phones_and_bert(text, language, version, final=False):
    # 1. 단일 언어 처리 (한국어, 일본어, 영어만 남김!)
    if language in {"en", "all_ja", "all_ko"}:
        language = language.replace("all_", "")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            # 일본어, 한국어는 그대로 통과
            formattext = text
            
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
            
        # 중국어 전용 정규식과 처리 로직은 깔끔하게 날려버렸다몬!
        phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
        
        # 한국어, 일본어, 영어는 기본적으로 zero bert를 사용함
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    # 2. 다국어 혼합 처리 (한국어, 일본어, 영어 믹스 모드)
    elif language in {"ja", "ko", "auto"}:
        textlist = []
        langlist = []
        # 필터에서도 중국어("zh")를 뺐다のだ! (한자가 섞여도 일본어/한국어 위주로 인식함)
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
                
        # 디버깅용 출력 (나중에 필요 없으면 지워도 돼のだ!)
        print("텍스트 분리:", textlist)
        print("언어 감지:", langlist)
        
        phones_list = []
        bert_list = []
        norm_text_list = []
        
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
            
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

    # 3. 에러 방지용 짧은 문장 보호막 (그대로 유지!)
    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text, language, version, final=True)

    # dtype이 정의된 전역변수라면 그대로 쓰고, 아니라면 bert.to(bert.dtype) 등으로 맞추면 된다몬
    return phones, bert.to(dtype), norm_text