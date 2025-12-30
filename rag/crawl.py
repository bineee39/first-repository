import os
import time
import json
import csv
from dotenv import load_dotenv
from openai import OpenAI
from playwright.sync_api import sync_playwright

# 1. 환경 변수 및 설정
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    print("❌ .env 파일에 'OPENAI_API_KEY'가 없습니다.")
    exit()

client = OpenAI(api_key=openai_api_key)

# --- [수정 포인트 1] 목표 개수를 200& C:/Users/user/anaconda3/envs/ds6_RAG/python.exe crawl.py으로 확실히 설정 ---
TARGET_CATEGORY_URL = "https://www.daisomall.co.kr/ds/exhCtgr/C208/CTGR_00014/CTGR_00057/CTGR_00366"
MAX_ITEMS = 209  
CSV_FILE = "daiso_analysis_result.csv"
HEADERS = ["상품명", "URL", "흡수력", "보습력", "자극도", "한줄요약"]

def get_product_links(page):
    """지연 로딩을 극복하고 200개의 링크를 수집할 때까지 정밀 스크롤합니다."""
    print(f"📂 카테고리 페이지 접속 중: {TARGET_CATEGORY_URL}")
    page.goto(TARGET_CATEGORY_URL, wait_until="networkidle")
    
    unique_links = set()
    prev_count = 0
    no_change_count = 0
    max_no_change = 7 # 7번 시도해도 안 늘어나면 진짜 끝임

    print(f"📜 목표 개수({MAX_ITEMS}개) 수집을 시작합니다. (20개 단위로 로딩됨)")

    while len(unique_links) < MAX_ITEMS:
        # 마우스 휠을 굴려 실제 사용자가 내리는 것처럼 시뮬레이션
        for _ in range(8): 
            page.mouse.wheel(0, 3000) 
            time.sleep(0.7)
        
        time.sleep(2.5) # 로딩 대기 시간 충분히 확보
        
        current_links = page.evaluate("""
            () => {
                const anchors = Array.from(document.querySelectorAll('a'));
                return anchors
                    .map(a => a.href)
                    .filter(href => href && href.includes('/pd/pdr/'));
            }
        """)
        
        for link in current_links:
            unique_links.add(link)
        
        current_count = len(unique_links)
        print(f"   🔄 현재 확보된 링크: {current_count}개 / {MAX_ITEMS}개")

        # 더 이상 안 늘어나는지 체크
        if current_count == prev_count:
            no_change_count += 1
            if no_change_count >= max_no_change:
                print("⚠️ 더 이상 새로운 상품이 없습니다. 수집을 종료합니다.")
                break
        else:
            no_change_count = 0
            
        prev_count = current_count

    final_links = list(unique_links)[:MAX_ITEMS]
    print(f"✅ {len(final_links)}개의 상품을 수집할 예정입니다.") # 이 메시지가 200으로 나와야 함!
    return final_links

def analyze_with_gpt(text):
    prompt = f"""
    당신은 화장품 리뷰 데이터 분석가입니다. 
    제공된 텍스트에서 제품의 '리뷰 통계' 수치와 '핵심 요약'을 추출하여 JSON 형식으로 답변하세요.
    항목: 흡수력, 보습력, 자극도, 한줄요약
    텍스트: {text[:8000]}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "JSON 형식으로만 답변하고 정보 없으면 '정보없음'으로 표시."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except:
        return {}

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True) 
        context = browser.new_context(user_agent="Mozilla/5.0...")
        page = context.new_page()

        # 링크 수집
        product_links = get_product_links(page)
        
        if not os.path.exists(CSV_FILE):
            with open(CSV_FILE, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=HEADERS)
                writer.writeheader()

        # 분석 및 저장
        for idx, link in enumerate(product_links):
            print(f"[{idx+1}/{len(product_links)}] 처리 중: {link}")
            try:
                page.goto(link, wait_until="domcontentloaded")
                time.sleep(2)
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                
                content = page.inner_text("body")
                title = page.title().replace("다이소몰", "").strip()
                analyzed = analyze_with_gpt(content)
                
                row = {
                    "상품명": title, "URL": link,
                    "흡수력": analyzed.get("흡수력", "정보없음"),
                    "보습력": analyzed.get("보습력", "정보없음"),
                    "자극도": analyzed.get("자극도", "정보없음"),
                    "한줄요약": analyzed.get("한줄요약", "정보없음")
                }

                with open(CSV_FILE, "a", encoding="utf-8-sig", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=HEADERS)
                    writer.writerow(row)
                print(f"   ✔️ 저장 완료: {row['한줄요약'][:30]}...")
            except Exception as e:
                print(f"   ❌ 오류: {e}")
            time.sleep(2.5)

        browser.close()
        print(f"🎉 완료! '{CSV_FILE}'를 확인하세요.")

if __name__ == "__main__":
    main()