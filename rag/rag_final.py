import streamlit as st
import pandas as pd
import os
import time
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------
# 1. 환경 설정 및 API 로드 (무손실 보존)
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path=env_path)

# ---------------------------------------------------------
# 2. 프리미엄 디자인 리뉴얼 (Daiso Red & Pink Review Box)
# ---------------------------------------------------------
st.set_page_config(page_title="다이소 뷰티 큐레이터 v62", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@300;400;600;800&display=swap');
    * { font-family: 'Pretendard', sans-serif; }

    .stApp { background-color: #FFFFFF; }
    
    /* 상단 배너 - 다이소 레드 프리미엄 그라데이션 */
    .welcome-banner {
        background: linear-gradient(135deg, #FF1535 0%, #D60E2A 100%);
        padding: 60px 20px; border-radius: 30px; color: white; text-align: center;
        margin-bottom: 40px; box-shadow: 0 20px 40px rgba(255, 21, 53, 0.15);
    }
    .welcome-banner h1 { margin: 0; font-size: 3.5rem; font-weight: 800; letter-spacing: -2px; }
    .welcome-banner p { margin: 20px 0 0; font-size: 1.2rem; font-weight: 300; opacity: 0.9; }

    /* 전문가 프로필 - 신뢰도 높은 스타일 */
    .expert-intro {
        display: flex; align-items: center; gap: 25px; background: #F8F9FA;
        padding: 35px; border-radius: 25px; border-left: 12px solid #FF1535;
        margin-bottom: 40px; box-shadow: 0 10px 20px rgba(0,0,0,0.03);
    }
    .expert-avatar {
        font-size: 55px; background: #FFFFFF; width: 95px; height: 95px;
        display: flex; align-items: center; justify-content: center;
        border-radius: 50%; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .expert-text { font-size: 1.2rem; color: #333; line-height: 1.7; font-weight: 500; }
    .expert-text b { color: #FF1535; font-weight: 700; }

    /* 채팅 메시지 카드 디자인 */
    [data-testid="stChatMessage"] {
        background-color: #fcfcfc; border: 1px solid #f0f0f0;
        border-radius: 25px; padding: 30px; margin-bottom: 25px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.02);
    }

    /* [v62.0 핵심] 베스트 리뷰 요약 핑크 박스 */
    .review-box {
        background-color: #FFF0F3; 
        border: 2px solid #FFD1DC; 
        border-radius: 20px; 
        padding: 25px;
        margin-top: 30px;
        margin-bottom: 20px;
        box-shadow: 0 8px 20px rgba(255, 21, 53, 0.05);
    }
    .review-title {
        font-weight: 800; color: #FF1535; font-size: 1.2rem;
        margin-bottom: 12px; display: flex; align-items: center; gap: 10px;
    }
    .review-content { color: #555; font-size: 1.1rem; line-height: 1.6; font-style: italic; }

    /* 구매 링크 버튼 스타일 */
    .buy-link {
        display: inline-block; padding: 15px 35px; background-color: #FF1535;
        color: white !important; text-decoration: none !important;
        border-radius: 50px; font-weight: 600; margin-top: 15px;
        box-shadow: 0 10px 20px rgba(255, 21, 53, 0.2);
        transition: all 0.3s ease;
    }
    .buy-link:hover { transform: translateY(-3px); box-shadow: 0 15px 25px rgba(255, 21, 53, 0.3); }

    /* 사이드바 디자인 */
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #eee; }
    .sidebar-title { color: #FF1535; font-weight: 800; font-size: 1.6rem; margin-bottom: 20px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. 데이터 로드 및 벡터 저장소 ([DIY패드 영구 삭제] 무손실)
# ---------------------------------------------------------
@st.cache_resource
def get_vectorstore():
    persist_directory = os.path.join(current_dir, "chroma_db_v62")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    csv_file = os.path.join(current_dir, "final_integrated_data_v2.csv")
    
    if not os.path.exists(csv_file):
        st.error("데이터 파일이 없습니다. 경로를 확인해주세요."); st.stop()
    
    df = pd.read_csv(csv_file, encoding="utf-8-sig")
    # [핵심] 규빈님이 지적하신 이상치 DIY패드 데이터 영구 삭제
    df = df[~df['상품명'].str.contains('DIY패드|diy패드|DIY 패드', na=False, case=False)]
    
    documents = []
    for _, row in df.iterrows():
        content = "\n".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
        documents.append(Document(
            page_content=content, 
            metadata={
                "상품명": str(row.get('상품명', '')),
                "review_count": int(row.get('리뷰수', 0)), 
                "rating": row.get('평점', 0),
                "detail": str(row.get('상세정보', '')),
                "summary": str(row.get('한줄요약', ''))
            }
        ))
    return Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_directory)

vectorstore = get_vectorstore()

# ---------------------------------------------------------
# 4-1. BM25 인덱스 생성 (무손실 보존)
# ---------------------------------------------------------
def simple_tokenizer(text):
    if not isinstance(text, str): return []
    return re.findall(r'[가-힣a-zA-Z0-9]{1,}', re.sub(r'[_\-().]', ' ', text))

@st.cache_resource
def get_bm25_index():
    df = pd.read_csv(os.path.join(current_dir, "final_integrated_data_v2.csv"), encoding="utf-8-sig")
    df = df[~df['상품명'].str.contains('DIY패드|diy패드', na=False, case=False)]
    doc_texts = [" ".join([str(row[c]) for c in ['상품명', '상세정보', '한줄요약'] if c in df.columns]) for _, row in df.iterrows()]
    return BM25Okapi([simple_tokenizer(d) for d in doc_texts]), doc_texts

bm25_index, bm25_documents = get_bm25_index()

# ---------------------------------------------------------
# 4-2. 최강 하이브리드 리랭킹 검색 (v52.0 로직 완전 사수)
# ---------------------------------------------------------
def get_advanced_context(query, k=60):
    query_clean = query.lower()
    v_docs = vectorstore.similarity_search(query_clean, k=k)
    
    doc_scores = {}
    for rank, doc in enumerate(v_docs):
        doc_scores[doc.page_content] = {'vector': 1.0 - (rank * 0.015), 'bm25': 0, 'doc': doc}
    
    if bm25_index:
        bm25_scs = bm25_index.get_scores(simple_tokenizer(query_clean))
        top_indices = sorted(range(len(bm25_scs)), key=lambda i: bm25_scs[i], reverse=True)[:k]
        max_v = max(bm25_scs) if max(bm25_scs) > 0 else 1
        for i in top_indices:
            text = bm25_documents[i]
            if text in doc_scores: doc_scores[text]['bm25'] = bm25_scs[i] / max_v
            else: doc_scores[text] = {'vector': 0, 'bm25': bm25_scs[i] / max_v, 'doc': Document(page_content=text)}

    # 키워드 사전
    category_keywords = {
        "토너": ["토너", "스킨", "미스트", "패드", "토너패드"],
        "크림": ["크림", "수딩크림", "젤크림", "보습크림", "영양크림", "멀티밤", "스틱"],
        "앰플": ["앰플", "세럼", "에센스", "오일앰플", "영양앰플"]
    }
    effect_keywords = {
        "진정": ["진정", "시카", "어성초", "수딩", "판테놀"],
        "미백": ["미백", "비타", "광채", "브라이트닝", "잡티"],
        "오일/고보습": ["오일", "영양", "고보습", "세라마이드", "밤", "스틱", "꾸덕", "심한 건성"]
    }

    target_cats = [k for k, v in category_keywords.items() if any(kw in query_clean for kw in v + [k])]
    target_effs = [k for k, v in effect_keywords.items() if any(kw in query_clean for kw in v + [k])]

    final_scored = []
    for text, sc in doc_scores.items():
        hybrid_score = (sc['vector'] * 0.7) + (sc['bm25'] * 0.3)
        prod_name = sc['doc'].metadata.get('상품명', '').lower()
        detail_text = sc['doc'].metadata.get('detail', '').lower()
        
        # 1. [계층 1] 카테고리 매칭 (+500,000점) - 최우선 순위
        if any(any(kw in prod_name for kw in category_keywords[cat]) for cat in target_cats):
            hybrid_score += 500000.0
        
        # 2. [계층 2] 효과/제형 정밀 매칭 (+100,000점)
        if any(any(kw in prod_name or kw in detail_text for kw in effect_keywords[eff]) for eff in target_effs):
            hybrid_score += 100000.0

        # 3. [계층 3] 리들샷 방어 페널티 로직 (v52.0 핵심 무손실)
        if any(kw in query_clean for kw in ["오일", "영양", "진정 토너"]):
            if any(kw in prod_name for kw in ["리들샷", "부스팅", "부스터"]):
                hybrid_score -= 800000.0
        
        # 4. [계층 4] 인기순 가중치
        review_count = sc['doc'].metadata.get('review_count', 0)
        hybrid_score += (review_count / 10) 
        
        final_scored.append((review_count, hybrid_score, sc['doc']))

    final_scored.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return final_scored[:3] # 상위 3개 제품만 결과로 생성

# ---------------------------------------------------------
# 5. [마스터 프롬프트] 가독성 & 형식 & 이모티콘 완벽 제어
# ---------------------------------------------------------
system_prompt = """귀하는 다이소의 전문 뷰티 큐레이터 '뷰티 버디'입니다. 

[🔍 규빈님 요청 답변 스타일 가이드 - v62.0]
1. **6단계 고정 포맷**: 반드시 아래의 라벨을 사용하여 답변하십시오.
   - 🧴 **피부 타입**: 사용자의 고민을 분석하고 제품의 적합성을 상세히 설명.
   - 🏷️ **카테고리**: 요청한 카테고리와의 일치성 확인.
   - 🔥 **인기순**: 실제 리뷰 수와 평점 언급.
   - 🧪 **성분 및 효과**: [핵심 성분], [작용 원리], [고민 해결]로 구분.
   - #REVIEW#: 본문에는 리뷰 라벨을 쓰지 말고 이 구분자 뒤에 리뷰 내용만 작성.
   - 🔗 **링크**: 구매 URL 명시.

2. **성분 섹션 가독성**: 
   - [핵심 성분], [작용 원리], [고민 해결] 사이에는 반드시 **줄바꿈(엔터) 두 번**을 넣어 가독성을 확보하십시오.
   - 데이터에 핵심 성분이 없으면 [핵심 성분] 타이틀과 내용을 아예 생략하십시오.

3. **기호 사용 금지**: 동그라미(○, ●, •), 별표, 빈 괄호([]) 등을 절대 사용하지 마십시오.

4. **전문성 유지**: 규빈님이 좋아하신 "지성 피부는 피지 분비가 많아..." 식의 깊이 있는 분석을 유지하십시오.

질문: {question}
참조 문서: {context}
답변:"""

prompt = PromptTemplate.from_template(system_prompt)
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)

# ---------------------------------------------------------
# 6. UI 렌더링 및 사이드바
# ---------------------------------------------------------
st.sidebar.markdown("<div class='sidebar-title'>💄 DAISO BEAUTY</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.write("✅ **v62.0 마스터 버전**")
st.sidebar.write("- 리들샷 페널티 로직 탑재")
st.sidebar.write("- 카테고리 50만점 가중치")
st.sidebar.write("- 베스트 리뷰 핑크 박스")
st.sidebar.markdown("---")

st.markdown("""
    <div class='welcome-banner'>
        <h1>DAISO BEAUTY CURATION</h1>
        <p>당신만을 위한 다이소 베스트셀러 정밀 큐레이션</p>
    </div>
    <div class='expert-intro'>
        <div class='expert-avatar'>💄</div>
        <div class='expert-text'>
            안녕하세요, 고객님! <b>다이소 전문 큐레이터</b>입니다.<br>
            고객님의 피부 고민을 해결할 <b>최적의 성분과 압도적 인기</b>를 가진 제품을 찾아드릴게요.
        </div>
    </div>
    """, unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"], unsafe_allow_html=True)

# ---------------------------------------------------------
# 7. 실행 로직 (스트리밍 & 리뷰 박스 분리 - 무손실)
# ---------------------------------------------------------
user_input = st.chat_input("피부타입과 고민을 입력하세요... (예: 건성인데 오일 앰플 추천)")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(f"<b>{user_input}</b>", unsafe_allow_html=True)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("다이소의 모든 데이터를 정밀 분석 중입니다..."):
            top_results = get_advanced_context(user_input)
            context = "\n\n".join([d[2].page_content for d in top_results])
            
            chain = ({"context": lambda x: context, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
            
            for chunk in chain.stream(user_input):
                full_response += chunk
                display_text = full_response.split("#REVIEW#")[0]
                message_placeholder.markdown(display_text + "▌", unsafe_allow_html=True)
            
            # 최종 렌더링
            if "#REVIEW#" in full_response:
                main_content, review_part = full_response.split("#REVIEW#")
                message_placeholder.markdown(main_content, unsafe_allow_html=True)
                
                # [v62.0 핵심] 베스트 리뷰 요약 핑크 박스 동적 출력
                st.markdown(f"""
                    <div class='review-box'>
                        <div class='review-title'>💬 베스트 리뷰 요약</div>
                        <div class='review-content'>"{review_part.strip()}"</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                message_placeholder.markdown(full_response, unsafe_allow_html=True)
                
    st.session_state.messages.append({"role": "assistant", "content": full_response})