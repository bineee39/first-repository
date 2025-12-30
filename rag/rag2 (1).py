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
# 1. 환경 설정
# ---------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv(dotenv_path=env_path)

# ---------------------------------------------------------
# 2. 프리미엄 디자인 및 UI/UX (다이소 레드 아이덴티티)
# ---------------------------------------------------------
st.set_page_config(page_title="다이소 뷰티 큐레이터", layout="wide")

st.markdown("""
    <style>
    /* 다이소 레드 그라데이션 배너 */
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

    /* 전문가 프로필 섹션 */
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

    /* 제품명 명시된 종합 리뷰 타원형 박스 */
    .summary-box {
        background-color: #F8F9FA;
        border: 1px solid #FF1535;
        border-radius: 100px;
        padding: 15px 35px;
        margin-top: 15px;
        margin-bottom: 25px;
        display: inline-flex;
        align-items: center;
        max-width: 95%;
        box-shadow: 2px 2px 10px rgba(255, 21, 53, 0.08);
    }
    .summary-label { 
        font-weight: 800; color: #FF1535; margin-right: 20px; min-width: 160px; 
        border-right: 2px solid #FF1535; padding-right: 15px; font-size: 0.95rem; text-align: center;
    }
    .summary-text { color: #333; font-size: 1rem; font-weight: 500; font-style: italic; }
    .section-title { font-weight: bold; color: #FF1535; margin: 30px 0 10px 0; font-size: 1.25rem; display: block; }
    
    /* 베스트 추천 박스 (골드 테마) */
    .best-box {
        background: linear-gradient(135deg, #FFF8E7 0%, #FFFAF0 100%);
        border: 2px solid #FFD700;
        border-radius: 15px;
        padding: 20px 30px;
        margin-top: 25px;
        display: block;
        max-width: 100%;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.25);
    }
    .best-title {
        font-weight: 900; color: #B8860B; font-size: 1.1rem; margin-bottom: 10px;
        display: flex; align-items: center; gap: 8px;
    }
    .best-text { color: #5D4E37; font-size: 1.05rem; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. 데이터 로드 및 벡터 저장소 
# ---------------------------------------------------------
@st.cache_resource
def get_vectorstore():
    persist_directory = os.path.join(current_dir, "chroma_db_v2")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small",chunk_size=1000)
    
    # 1. 이미 저장된 DB가 있는지 확인
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    # 2. DB가 없으면 새로 생성
    documents = []
    csv_file = os.path.join(current_dir, "final_integrated_data_v2.csv")
    
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, encoding="utf-8-sig")
        
        for _, row in df.iterrows():
            content_parts = []
            for col in df.columns:
                val = row[col]
                if pd.notna(val):
                    content_parts.append(f"{col}: {val}")
            page_content = "\n".join(content_parts)
            
            documents.append(Document(
                page_content=page_content, 
                metadata={"source": csv_file, "row": _}
            ))
    
    if not documents:
        st.error("데이터 파일을 찾을 수 없습니다.")
        st.stop()
    
    # 배치 처리로 벡터 저장소 생성 (토큰 제한 우회)
    vectorstore = Chroma.from_documents(
        documents=documents[:50],
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # 나머지 추가
    batch_size = 50
    for i in range(50, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        vectorstore.add_documents(batch)
    
    return vectorstore

# API 키 확인
if "OPENAI_API_KEY" not in os.environ:
    api_key = st.sidebar.text_input("OpenAI API Key를 입력하세요", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        st.sidebar.warning("🔑 API Key를 입력해 주세요.")
        st.stop()

vectorstore = get_vectorstore()

# ---------------------------------------------------------
# 4-1. BM25 인덱스 생성
# ---------------------------------------------------------
def simple_tokenizer(text):
    if not isinstance(text, str): return []
    text = re.sub(r'[_\-().]', ' ', text)
    tokens = re.findall(r'[가-힣a-zA-Z0-9]{1,}', text) 
    return tokens

@st.cache_resource
def get_bm25_index():
    """BM25 검색을 위한 인덱스 생성"""
    csv_file = os.path.join(current_dir, "final_integrated_data_v2.csv")
    
    if not os.path.exists(csv_file):
        return None, None
    
    df = pd.read_csv(csv_file, encoding="utf-8-sig")
    
    # 문서 텍스트 생성
    documents = []
    doc_texts = []
    
    for _, row in df.iterrows():
        content_parts = []
        # 검색에 유용한 컬럼만 결합
        cols_to_use = ['상품명', '상세정보', '한줄요약']
        for col in cols_to_use:
            if col in df.columns:
                val = row[col]
                if pd.notna(val):
                    content_parts.append(str(val))
        text = " ".join(content_parts)
        documents.append(text)
        doc_texts.append(text)
    
    # 토큰화
    tokenized_docs = [simple_tokenizer(doc) for doc in doc_texts]
    
    # BM25 인덱스 생성
    bm25 = BM25Okapi(tokenized_docs)
    
    return bm25, documents

bm25_index, bm25_documents = get_bm25_index()

# ---------------------------------------------------------
# 4-2. 쿼리 리라이팅 함수
# ---------------------------------------------------------
@st.cache_data(ttl=3600)  # 1시간 캐싱
def rewrite_query(original_query: str) -> str:
    """사용자 질문을 검색에 최적화된 형태로 재작성"""
    rewrite_prompt = f"""사용자의 질문을 화장품 검색에 최적화된 키워드로 재작성하세요.

[규칙]
1. 구어체를 검색 키워드로 변환
2. 오타가 있으면 교정 (어선초→어성초, 시카그림→시카크림)
3. 계절/상황을 효능 키워드로 변환 (겨울→보습, 건조 방지)
4. 성별 표현을 명확히 (남친→남성용, 남자)
5. 원래 질문의 핵심 의도를 유지
6. **제품명 키워드는 띄어쓰기 버전과 붙여쓰기 버전 모두 포함**
   예: "PDRN 콜라겐 토너" → "PDRN 콜라겐 토너 PDRN콜라겐토너"
7. 재작성된 질문만 출력 (설명 없이)

원래 질문: {original_query}
재작성된 질문:"""
    
    try:
        rewrite_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        result = rewrite_llm.invoke(rewrite_prompt)
        rewritten = result.content.strip()
        return rewritten if rewritten else original_query
    except:
        return original_query

# ---------------------------------------------------------
# 4-3. 카테고리 키워드 정의 및 추출 함수
# ---------------------------------------------------------
CATEGORY_KEYWORDS = {
    # 제품 유형 키워드 (질문에서 추출할 키워드: 상품명에서 매칭할 키워드들)
    "패드": ["패드", "pad"],
    "토너": ["토너", "스킨", "toner"],
    "크림": ["크림", "cream"],
    "로션": ["로션", "lotion"],
    "세럼": ["세럼", "에센스", "앰플", "serum", "essence", "ampoule"],
    "클렌징": ["클렌징", "클렌저", "폼", "워시", "cleansing", "cleanser", "foam", "wash"],
    "마스크": ["마스크", "팩", "mask", "pack"],
    "미스트": ["미스트", "스프레이", "mist", "spray"],
    "오일": ["오일", "oil"],
    "선크림": ["선크림", "선블록", "자외선", "sunscreen", "sun block", "spf"],
    "립": ["립", "lip"],
    "아이": ["아이크림", "아이세럼", "eye"],
}

def extract_category_from_query(query: str) -> list:
    """질문에서 제품 카테고리 키워드 추출"""
    query_lower = query.lower()
    found_categories = []
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        # 질문에 카테고리 키워드가 포함되어 있는지 확인
        if category in query_lower:
            found_categories.append(category)
        # 동의어도 체크
        for kw in keywords:
            if kw in query_lower and category not in found_categories:
                found_categories.append(category)
                break
    
    return found_categories

# ---------------------------------------------------------
# 4-4. 성분 키워드 정의 및 추출 함수
# ---------------------------------------------------------
INGREDIENT_KEYWORDS = [
    # 주요 화장품 성분 목록
    "어성초", "시카", "마데카", "센텔라", "티트리", "녹차", "병풀",
    "히알루론", "히알루론산", "콜라겐", "레티놀", "비타민", "나이아신아마이드",
    "세라마이드", "펩타이드", "글루타치온", "아르부틴", "알파하이드록시",
    "살리실산", "aha", "bha", "pha", "pdrn", "프로폴리스",
    "스쿠알란", "호호바", "아르간", "로즈힙", "쌀", "꿀", "달팽이",
    "알로에", "카밍", "칼라민", "진정", "수분", "보습",
]

def extract_ingredients_from_query(query: str) -> list:
    """질문에서 성분 키워드 추출"""
    query_lower = query.lower()
    found_ingredients = []
    
    for ingredient in INGREDIENT_KEYWORDS:
        if ingredient.lower() in query_lower:
            found_ingredients.append(ingredient.lower())
    
    return found_ingredients

def doc_matches_ingredients(doc_content: str, ingredients: list) -> bool:
    """문서가 지정된 성분을 포함하는지 확인 (상품명 또는 상세정보)"""
    if not ingredients:
        return True  # 성분 지정 없으면 모든 문서 통과
    
    doc_lower = doc_content.lower()
    
    # 모든 요청 성분이 문서에 포함되어야 함 (AND 조건)
    for ingredient in ingredients:
        if ingredient not in doc_lower:
            return False
    
    return True

# ---------------------------------------------------------
# 4-5. 피부 타입 키워드 정의 및 추출 함수
# ---------------------------------------------------------
SKIN_TYPE_KEYWORDS = {
    "건성": ["건성", "건조", "dry"],
    "지성": ["지성", "오일리", "oily", "피지"],
    "복합성": ["복합성", "combination"],
    "민감성": ["민감", "sensitive", "약산성"],
    "트러블": ["트러블", "여드름", "acne", "trouble"],
}

def extract_skin_types_from_query(query: str) -> list:
    """질문에서 피부 타입 키워드 추출"""
    query_lower = query.lower()
    found_types = []
    
    for skin_type, keywords in SKIN_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower and skin_type not in found_types:
                found_types.append(skin_type)
                break
    
    return found_types

def doc_matches_skin_types(doc_content: str, skin_types: list) -> bool:
    """문서가 지정된 피부 타입에 적합한지 확인 (상세정보에서 확인)"""
    if not skin_types:
        return True  # 피부 타입 지정 없으면 모든 문서 통과
    
    doc_lower = doc_content.lower()
    
    # 요청한 피부 타입 중 하나라도 매칭되면 통과 (OR 조건)
    for skin_type in skin_types:
        if skin_type in SKIN_TYPE_KEYWORDS:
            for keyword in SKIN_TYPE_KEYWORDS[skin_type]:
                if keyword in doc_lower:
                    return True
    
    return False

def doc_matches_category(doc_content: str, categories: list) -> bool:
    """문서가 지정된 카테고리에 해당하는지 확인 (상품명에서만 매칭)"""
    if not categories:
        return True  # 카테고리 지정 없으면 모든 문서 통과
    
    # 상품명 추출 (상품명에서만 카테고리 확인)
    product_name = ""
    name_match = re.search(r'상품명:\s*([^\n]+)', doc_content)
    if name_match:
        product_name = name_match.group(1).lower()
    
    if not product_name:
        return False  # 상품명을 찾을 수 없으면 제외
    
    for category in categories:
        if category in CATEGORY_KEYWORDS:
            for keyword in CATEGORY_KEYWORDS[category]:
                # 상품명에 카테고리 키워드가 있는지만 확인 (엄격한 필터링)
                if keyword in product_name:
                    return True
    
    return False

# ---------------------------------------------------------
# 4-6. 하이브리드 검색 (쿼리 리라이팅 + 벡터 + BM25 + 카테고리/성분/피부타입 필터링)
# ---------------------------------------------------------
def get_advanced_context(query, k=15):
    """
    하이브리드 검색: 쿼리 리라이팅 + 벡터(70%) + BM25(30%)
    + 카테고리 필터링 + 성분 필터링 + 피부 타입 필터링 + 리뷰수 우선 정렬
    """
    
    # 0. 질문에서 카테고리, 성분, 피부 타입 추출
    requested_categories = extract_category_from_query(query)
    requested_ingredients = extract_ingredients_from_query(query)
    requested_skin_types = extract_skin_types_from_query(query)
    
    # 1. 쿼리 리라이팅
    rewritten_query = rewrite_query(query)
    
    # 필터링을 위해 더 많은 후보군 확보 (10배수)
    has_filters = requested_categories or requested_ingredients or requested_skin_types
    retrieval_k = k * 10 if has_filters else k
    
    # 2. 벡터 검색 (Dense)
    vector_docs = vectorstore.similarity_search(rewritten_query, k=retrieval_k)
    
    # 3. BM25 검색 (Sparse)
    bm25_results = []
    if bm25_index and bm25_documents:
        tokenized_query = simple_tokenizer(rewritten_query)
        bm25_scores = bm25_index.get_scores(tokenized_query)
        
        # 상위 k개 인덱스
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:retrieval_k]
        bm25_results = [(bm25_documents[i], bm25_scores[i]) for i in top_indices]
    
    # 4. 하이브리드 점수 계산
    doc_scores = {}  # {문서내용: (벡터점수, BM25점수, 원본문서)}
    
    # 벡터 결과 추가 (순위 기반 점수: 1위=1.0, 2위=0.9...)
    for rank, doc in enumerate(vector_docs):
        vector_score = 1.0 - (rank * 0.02)  # 더 세밀한 점수 차이
        doc_scores[doc.page_content] = {
            'vector': vector_score,
            'bm25': 0,
            'doc': doc
        }
    
    # BM25 결과 병합
    if bm25_results:
        max_bm25 = max(score for _, score in bm25_results) if bm25_results else 1
        for doc_text, score in bm25_results:
            normalized_bm25 = score / max_bm25 if max_bm25 > 0 else 0
            if doc_text in doc_scores:
                doc_scores[doc_text]['bm25'] = normalized_bm25
            else:
                doc_scores[doc_text] = {
                    'vector': 0,
                    'bm25': normalized_bm25,
                    'doc': Document(page_content=doc_text)
                }
    
    # 5. 카테고리 + 성분 + 피부 타입 필터링 + 점수 계산
    final_scored = []
    for doc_text, scores in doc_scores.items():
        doc = scores['doc']
        
        # ★ 카테고리 필터링: 요청한 카테고리에 매칭되는 문서만 통과
        if not doc_matches_category(doc_text, requested_categories):
            continue
        
        # ★ 성분 필터링: 요청한 성분이 포함된 문서만 통과
        if not doc_matches_ingredients(doc_text, requested_ingredients):
            continue
        
        # ★ 피부 타입 필터링: 요청한 피부 타입에 적합한 문서만 통과
        if not doc_matches_skin_types(doc_text, requested_skin_types):
            continue
        
        hybrid_score = (scores['vector'] * 0.7) + (scores['bm25'] * 0.3)
        
        # 리뷰수 추출
        review_count = 0
        match = re.search(r'리뷰수:\s*(\d+)', doc.page_content)
        if match:
            review_count = int(match.group(1))
        
        final_scored.append((review_count, hybrid_score, doc))
    
    # 6. 정렬: 리뷰수 1순위, 하이브리드 점수 2순위
    final_scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    # 필터링 결과가 없으면 필터링 없이 재시도 (Smart Pivot)
    if not final_scored and has_filters:
        for doc_text, scores in doc_scores.items():
            doc = scores['doc']
            hybrid_score = (scores['vector'] * 0.7) + (scores['bm25'] * 0.3)
            review_count = 0
            match = re.search(r'리뷰수:\s*(\d+)', doc.page_content)
            if match:
                review_count = int(match.group(1))
            final_scored.append((review_count, hybrid_score, doc))
        final_scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    return "\n\n".join([d[2].page_content for d in final_scored[:6]])

# ---------------------------------------------------------
# 5. 마스터 프롬프트 (자연스러운 대화체 + Smart Pivot)
# ---------------------------------------------------------
system_prompt = """너는 다이소의 전문 뷰티 큐레이터 '뷰티 버디'입니다. 

[🔍 키워드 매칭 원칙 - 최우선]
질문에 포함된 **핵심 명사(제품군, 성분, 브랜드명 등)**가 모두 제품명이나 상세정보에 포함된 제품만 추천하십시오.
- 예: "어성초 패드" → "어성초"와 "패드"가 모두 포함된 제품만 추천
- 예: "시카 크림" → "시카"와 "크림"이 모두 포함된 제품만 추천
- 예: "마데카 세럼" → "마데카"와 "세럼"이 모두 포함된 제품만 추천

[필수 준수 사항]
1. **카테고리 명시 질문** (예: "미스트 추천해줘")
   - 요청한 제품군과 일치하는 제품을 우선 추천
   - 정확히 없으면 유사 제품(수분크림 → 크림, 젤) 추천 가능

2. **효능 기반 질문** (예: "여드름 흉터 없애고 싶어", "피부 뒤집어진 거 진정시킬 거")
   - '상세정보'와 '한줄요약'에서 관련 효능 찾아 추천
   - 제품군 관계없이 추천 가능

3. **리뷰수/인기 기반 질문** (예: "가장 리뷰 많은 어성초 패드")
   - 리뷰수가 가장 많은 제품을 정확히 찾아 "리뷰수가 XX개로 가장 많으며" 명시

4. **성별/선물 관련 질문** (예: "남친 선물용")
   - "남자", "남성" 텍스트가 포함된 제품 우선 추천
   - 없으면 "남성분들도 사용하기 좋은" 간편한 제품 추천, "선물용" 포인트 언급

5. **계절 관련 질문** (예: "겨울에 쓸만한 거")
   - 겨울 → 보습, 건조 방지 / 여름 → 산뜻함, 피지 조절
   - 계절과 연관된 효능 자연스럽게 언급

6. **대안 제시 (Smart Pivot)**
   - 요청 제품이 없어도 "없다"고 하지 말고 논리적 대안 제시
   - "데이터 부족", "정보 없음" 같은 부정적 표현 금지

[답변 구조]
각 제품 추천 시:
1. **상품명 소개**: "추천드리는 제품은 [상품명](URL)입니다."
2. **상세 설명**: 핵심 성분, 효능을 자연스럽게 설명
3. **사용자 반응**: 흡수력, 보습력, 자극도 수치를 문장에 녹여서 설명
4. **신뢰도**: 평점과 리뷰수 언급
   <div class='section-title'>✨ 구매자 종합 리뷰 분석</div>
   <div class='summary-box'><span class='summary-label'>💡 [상품명] 리뷰 요약</span><span class='summary-text'>"[한줄요약]"</span></div>


**종합 결론** (2개 이상 추천 시에만, 1개만 추천한 경우 생략):
[베스트 픽 선정 기준]
1순위: 질문의 핵심 고민에 가장 적합한 제품
2순위: 리뷰수가 많고 평점이 높은 제품
3순위: 사용자 만족도(흡수력, 보습력, 자극도)가 높은 제품

<div class='best-box'>
  <div class='best-title'>🏆 베스트 PICK</div>
  <div class='best-text'>[베스트 제품명]을 가장 추천드려요! [추천 이유 한 줄]</div>
</div>
질문: {question}
참조 문서: {context}
답변:"""

prompt = PromptTemplate.from_template(system_prompt)
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)

# ---------------------------------------------------------
# 6. UI 렌더링: 레드 테마 배너 및 챗봇 캐릭터
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
            오늘은 어떤 세련된 변화를 꿈꾸고 계신가요? 피부 고민을 말씀해 주시면 최적의 제품 라인업을 제안해 드릴게요. 😊
        </div>
    </div>
    """, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"], unsafe_allow_html=True)

# ---------------------------------------------------------
# 7. 실행 로직: 스트리밍 + 무손실 기능
# ---------------------------------------------------------
user_input = st.chat_input("피부 고민이나 궁금한 제품을 말씀해주세요!")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("데이터에서 최적의 대안을 분석 중입니다..."):
            context = get_advanced_context(user_input)
            chain = (
                {"context": lambda x: context, "question": RunnablePassthrough()}
                | prompt | llm | StrOutputParser()
            )
            
            for chunk in chain.stream(user_input):
                full_response += chunk
                time.sleep(0.015)
                message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            
            message_placeholder.markdown(full_response, unsafe_allow_html=True)
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})
