# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 김규빈님
- 리뷰어 : 김지수


# PRT(Peer Review Template)

- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**

텍스트 입력:i want to go sleep
인코딩 결과: [2, 117, 4, 106, 736]   
    
- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**

   ```   
   딕셔너리를 빈도수순으로 내림차순 정렬하고 ,
   ```
   storage_items = sorted(storage.items(), key=lambda x: (-x[1], x[0]))
   
   ```
   저희 팀도 이번 프로그래밍을 통해 lamda 라는 용어를 알게되었습니다.
   그래서 가장 반가웠고, 핵심적인 부분이라고 생각했습니다.
   ```     
- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
   
   with open('Avengers.txt', 'r') as f:
    Avengers_text = f.read()
    print("파일을 성공적으로 불러왔습니다.")

   ```
   따로 기록으로 남겨놓으신건 아니지만,
   규빈님이 코더해주실 때, 'Aengers.txt' 파일의 'A'부분이 대문자인걸 인지하지 못하시고
   뒤늦게 소문자로 진행 중이었다는 걸 깨달아서 생각보다 오래걸렸다는 말씀을 해주셨습니다.
   ``` 
        
- [x]  **4. 회고를 잘 작성했나요?**
   ```
   생각보다 프로그래밍이 오래걸려서 회고 작성을 따로 하지 못하셨다하여,
   느낀점을 따로 말씀해주셨습니다.
   규빈님께서는 앞서 말씀해주신 에러 확인을 빨리 하지 못해 어려움을 느낀 부분이 아쉬웠고,
   순탄할 줄 알았던 내림차순 부분이 생각보다 오래걸렸지만,
   같이 진행한 그루님깨서 lamda 사용을 제안해주셔서 흥미롭다는 말씀을 해주셨습니다.
   ```
        
- [x]  **5. 코드가 간결하고 효율적인가요?**
    
    with open("Avengers.txt", "r", encoding="utf-8") as f:

def preprocess(text:str) --> list[str]:
    text = text. lower()
    text =

    return tokens
     
The Tesseract has awakened.
It is on a neutral world, a human world.
They wield its power,
but our ally knows it's working, so that they never will learn.
He's ready to lead, and our force, our Chitauri will follow.
The world will be his and the universe yours.
And the humans, what can they do, but burn.
How bad is it?
That's the problem, sir. We don't know.
Dr. Selvig read an energy surge from the Tesseract four hour ago.
NASA didn't authorize Selvig to pull the test phase.
He wasn't testing it

import re
from collections import Counter
     

with open('Avengers.txt', 'r') as f:
    Avengers_text = f.read()
    print("파일을 성공적으로 불러왔습니다.")
#텍스트 전처리 과정
def preprocess(text):
    text =text.lower()
    new_text=[]
    for word in text:
      if word.isalpha():
        new_text.append(word)
      else:
        new_text.append(' ')
    new_text=''.join(new_text)
    final_word=new_text.split()
    return final_word

#각 단어 별 빈도수를 딕셔너리 형태로 저장
storage={}
all_text=preprocess(Avengers_text)
for i in all_text:
  if i in storage:
    storage[i] +=1
  else:
    storage[i]=1

# 딕셔너리를 빈도수순으로 내림차순 정렬하고 ,
storage_items = sorted(storage.items(), key=lambda x: (-x[1], x[0]))

# 정렬 순서대로 정수 인텍스 부여
vocab = {word: idx for idx, (word, _) in enumerate(storage_items)}

# 텍스트를 input()으로 입력받아서 정수를 return 하는 함수를 만든다.
def encoder_from_input(vocab):
    s = input("텍스트 입력:")
    toks = preprocess(s)
    return [vocab[t] for t in toks if t in vocab]

print("--- 생성된 단어 사전 (vocab) ---")
print(vocab)
print("-" * 25)
encoded_result = encoder_from_input(vocab)
print("인코딩 결과:", encoded_result)



# 회고(참고 링크 및 코드 개선)

# 딕셔너리를 빈도수순으로 내림차순 정렬하고 ,
storage_items = sorted(storage.items(), key=lambda x: (-x[1], x[0]))
```
규빈님의 코드를 보면서 저희 팀과 비슷한 코드가 있다는 것을 발견하고 굉장히 반가웠습니다.
```
```
프로그래밍을 하면서 자잘한 문제들이 있어서 어려움이 있다고 하셨는데,
그만큼 깔끔하고 너무 어렵지 않게 분석 가능한 프로그래밍을 하신 것 같아 그 부분이 인상적이었습니다.
```
