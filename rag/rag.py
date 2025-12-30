import streamlit as st
import pandas as pd
import os
import time
import re
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---------------------------------------------------------
# 1. 프리미엄 디자인 및 UI/UX (가독성 & 다이소 레드 무손실)
# ---------------------------------------------------------
st.set_page_config(page_title="다이소 뷰티 큐레이터", layout="wide")

st.markdown("""
    <style>
    /* 다이소 레드 그라데이션 웰컴 배너 */
    .welcome-banner {
        background: linear-gradient(135deg, #FF1535 0%, #FF4D6D 100%);
        padding: 45px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 35px;
        box-shadow: 0 10px 30px rgba(255, 21, 53, 0.25);
    }
    .welcome-banner h1 { margin: 0; font-size: 2.8rem; font-weight: 900; letter-spacing: -1px; }
    .welcome-banner p { margin: 15px 0 0; font-size: 1.2rem; font-weight: 300; opacity: 0.95; }

    /* 전문가 프로필 섹션 (캐릭터 배치) */
    .expert-intro {
        display: flex;
        align-items: center;
        gap: 25px;
        background-color: white;
        padding: 25px;
        border-radius: 20px;
        border: 1px solid #f1f3f5;
        margin-bottom: 30px;
    }
    .expert-avatar {
        font-size: 50px;
        background: #FFF0F3;
        width: 85px;
        height: 85px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        border: 2px solid #FF1535;
    }
    .expert-text { font-size: 1.2rem; color: #2d3436; font-weight: 600; line-height: 1.6; }

    /* 제품명이 명시된 종합 리뷰 타원형(Pill) 박스 */
    .summary-box {
        background-color: #F8F9FA;
        border: 1px solid #FF1535;
        border-radius: 100px;
        padding: 15px 35px;
        margin-top: 15px;
        display: inline-flex;
        align-items: center;
        max-width: 95%;
        box-shadow: 2px 2px 10px rgba(255, 21, 53, 0.08);
    }
    .summary-label { 
        font-weight: 800; color: #FF1535; margin-right: 20px; 
        min-width: 180px; border-right: 2px solid #FF1535; padding-right: 15px; 
        font-size: 0.9rem; text-align: center;
    }
    .summary-text { color: #333; font-size: 1rem; font-weight: 500; font-style: italic; }
    .section-title { font-weight: bold; color: #FF1535; margin: 30px 0 10px 0; font-size: 1.25rem; display: block; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 데이터 로드 및 벡터 저장소 (FAISS 무손실 로직)
# ---------------------------------------------------------
@st.cache_resource
def get_vectorstore():
    documents = []
    csv_file = "final_integrated_data.csv"
    if os.path.exists(csv_file):
        documents.extend(CSVLoader(file_path=csv_file, encoding="utf-8").load())
    if os.path.exists("text_data"):
        documents.extend(DirectoryLoader("text_data", glob="*.txt", loader_cls=lambda path: TextLoader(path, encoding="utf-8")).load())
    
    if not documents:
        st.error("데이터 파일이 없습니다! final_integrated_data.csv 파일을 확인해주세요.")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    return FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())

if "OPENAI_API_KEY" not in os.environ:
    api_key = st.sidebar.text_input("OpenAI API Key를 입력하세요", type="password")
    if api_key: os.environ["OPENAI_API_KEY"] = api_key
    else: st.sidebar.warning("🔑 API Key를 입력해주세요."); st.stop()

vectorstore = get_vectorstore()

# ---------------------------------------------------------
# 3. 가중치 기반 지능형 검색 (Full Scoring Logic 완벽 보존)
# ---------------------------------------------------------
CATEGORY_MAP = {
    "클렌징": ["클렌징", "폼", "오일", "워터", "클렌저", "세안"],
    "토너": ["토너", "스킨", "미스트", "패드", "물스킨"],
    "세럼": ["세럼", "앰플", "에센스", "부스팅", "도입액"],
    "크림": ["크림", "수분크림", "모이스처라이저", "로션"]
}
EFFECT_MAP = {"탄력": ["탄력", "리프팅"], "진정": ["진정", "시카", "어성초", "뒤집", "예민"], "보습": ["보습", "수분", "속건조"]}

def get_advanced_context(query, k=15):
    docs = vectorstore.similarity_search(query, k=k)
    target_cats = [cat for cat, syns in CATEGORY_MAP.items() if any(s in query for s in syns)]
    target_effs = [eff for eff, syns in EFFECT_MAP.items() if any(s in query for s in syns)]
    
    scored_docs = []
    for doc in docs:
        content = doc.page_content.lower()
        score = 0
        if any(any(s in content for s in CATEGORY_MAP[c]) for c in target_cats): score += 12
        if any(any(s in content for s in EFFECT_MAP[e]) for e in target_effs): score += 18
        if any(x in content for x in ["리뷰", "평점", "요약", "한줄"]): score += 8
        scored_docs.append((score, doc))
    
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return "\n\n".join([d[1].page_content for d in scored_docs[:6]])

# ---------------------------------------------------------
# 4. [마스터 프롬프트] 범용 논리 엔진 + 오타 박멸 + 무손실 규칙
# ---------------------------------------------------------
system_prompt = """귀하는 다이소 매장에서 판매되는 기초 화장품만을 전문적으로 취급하는 고품격 뷰티 큐레이터 '뷰티 버디'입니다. 

[🧠 피부 컨디션별 자극 차단 논리]
1. 사용자가 "뒤집어졌다", "따갑다", "붉어졌다", "예민하다"고 할 경우, **리들샷(Riddleshot)**처럼 미세 침이 포함된 제품이나 기능성 고함량/각질 제거 제품은 절대 추천하지 마십시오. 
2. 이러한 손상된 피부에는 오직 **비자극성 진정(시카, 어성초, 판테놀)** 제품만 추천하십시오.

[🚫 용어 창조 및 환각 절대 엄금]
1. **슬리퍼리 트리트먼트**, **드레싱 마무리** 등 문서에 없는 해괴한 용어를 절대 지어내지 마십시오.
2. 오직 제공된 [참조 문서]에 기재된 팩트만을 바탕으로 답변하십시오.

[🚫 OCR 오타 실시간 교정 - 절대 엄격 준수]
참조 문서나 질문에 오타가 있어도 답변은 반드시 **표준 정석 명칭**으로 교정하십시오.
- **문광미백 -> 윤광미백**, **리들샛/리들셋 -> 리들샷**, **에이 에이치 씨 -> AHC**
- **타이팅 -> 타이트닝**, **프름폴리스 -> 프로폴리스**, **클랜징 -> 클렌징**
- 제품명에 '문광'이 보이면 100% '윤광'으로 고치고, '리들샛'은 100% '리들샷'으로 고쳐서 출력하십시오.

[🚫 루틴 카테고리 매칭 및 거절 지침]
1. 루틴 단계에 맞는 용도의 제품만 매칭하십시오. (클렌징 단계에 앰플 추천 절대 금지)
2. 성분 미함유 시: "데이터 내에서 해당 성분을 찾기 어렵다"고 정중히 사과하고 끝내십시오.
3. 서비스 외 질문(색조, 위치 등): 정중히 역할을 밝히고 거절 멘트만 남기고 깔끔하게 종료하십시오.

[⚠️ 답변 구성 및 UI]
- 상세 설명: [성분] -> [효과] -> [해결] 흐름 및 **줄바꿈(엔터)** 활용.
- 리뷰 UI: 반드시 제품명을 포함하여 아래 구조로 **단 한 번만** 출력하십시오.
<div class='section-title'>✨ 구매자 종합 리뷰 분석</div>
<div class='summary-box'><span class='summary-label'>💡 [제품명] 리뷰 요약</span><span class='summary-text'>"[한줄요약내용]"</span></div>

질문: {question}
참조 문서: {context}
답변:"""

prompt = PromptTemplate.from_template(system_prompt)
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)

# ---------------------------------------------------------
# 5. UI 렌더링: 레드 테마 배너 및 챗봇 캐릭터
# ---------------------------------------------------------
st.markdown("""
    <div class='welcome-banner'>
        <h1>THE DAISO BEAUTY CURATION</h1>
        <p>다이소의 붉은 열정으로 고객님의 아름다움을 가장 세련되게 큐레이팅합니다.</p>
    </div>
    <div class='expert-intro'>
        <div class='expert-avatar'>💄</div>
        <div class='expert-text'>
            반갑습니다, 고객님. 다이소 뷰티 큐레이터입니다.<br>
            고객님의 <b>현재 피부 컨디션</b>에 가장 안전하고 효과적인 실제 입점 제품들을 제안해 드릴게요. 😊
        </div>
    </div>
    """, unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"], unsafe_allow_html=True)

# ---------------------------------------------------------
# 6. 실행 로직: 스트리밍(속도 조절) & 무손실 기능 수행
# ---------------------------------------------------------
user_input = st.chat_input("피부 고민이나 궁금한 제품을 말씀해주세요!")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        context = get_advanced_context(user_input)
        chain = ({"context": lambda x: context, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
        
        for chunk in chain.stream(user_input):
            full_response += chunk
            time.sleep(0.015) 
            message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
        message_placeholder.markdown(full_response, unsafe_allow_html=True)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})