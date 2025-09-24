# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 김규빈님
- 리뷰어 : 조은별

# PRT(Peer Review Template)
- [Y]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
        - 중요! 해당 조건을 만족하는 부분을 캡쳐해 근거로 첨부
    
import random

class Account:
    account_count = 0
    #account_count: 지금까지 생성된 총 계좌 개수 추적용
    def __init__(self, owner_name, balance):
        #owner_name (str): 계좌 소유주 이름.
        #balance (int): 초기 잔액.
        # 계좌 기본 정보 설정(은행이름은 모두은행으로 통일)
        self.bank_name = "모두은행"
        self.owner_name = owner_name
        #초기 잔액 설정
        if balance < 0:
            raise ValueError("초기 잔액은 0 이상이어야 합니다.")
        self.balance = balance

        # 2 계좌번호 랜덤 생성 (3자리-2자리-6자리)
        num1 = f"{random.randint(0, 999):03d}"
        num2 = f"{random.randint(0, 99):02d}"
        num3 = f"{random.randint(0, 999999):06d}"
        self.account_number = f"{num1}-{num2}-{num3}"

        # 3 계좌 개수 증가(객체 만들때마다+1)
        Account.account_count += 1

        # 4 거래 관련 변수 초기화
        self.transaction_history = []
        self.deposit_count = 0
        self.transaction_num = 0

        print(f"[{self.owner_name}님의 계좌가 개설되었습니다]")
        print(f"은행명: {self.bank_name}")
        print(f"계좌번호: {self.account_number}")
        print(f"초기잔액: {self.balance:,}원\n")


    def deposit(self, amount):
        #입금함수
        if amount < 1:
            print("입금은 1원 이상만 가능합니다.")
            return
        #입금횟수,거래횟수 체크
        self.balance += amount
        self.deposit_count += 1
        self.transaction_num += 1

        log_msg = f"{self.transaction_num}회: 입금\t금액: {amount:<7,}\t잔액: {self.balance:,}"
        self.transaction_history.append(log_msg)
        print(log_msg)

        # 5회 입금마다 1% 이자 지급
        if self.deposit_count % 5 == 0:
            interest = int(self.balance * 0.01)
            self.balance += interest
            self.transaction_num += 1

            interest_msg = f"{self.transaction_num}회: 이자지급\t금액: {interest:<7,}\t잔액: {self.balance:,}"
            self.transaction_history.append(interest_msg)
            print(interest_msg)


    def withdraw(self, amount):
        #출금 함수
        if amount > self.balance:
            print("오류: 잔액이 부족합니다.")
            return
        #잔고-출금액,거래횟수+1
        self.balance -= amount
        self.transaction_num += 1

        log_msg = f"{self.transaction_num}회: 출금\t금액: {amount:<7,}\t잔액: {self.balance:,}"
        self.transaction_history.append(log_msg)
        print(log_msg)

    #class가 하는 일
    @classmethod
    def get_account_num(cls):
        #생성된 총 계좌 개수 출력
        print(f"\n총 생성된 계좌의 개수: {cls.account_count}개")


    def history(self):
        #계좌의 입.출금 내역을 모두 출력
        print("\n------ 거래 내역 ------")
        if not self.transaction_history:
            print("거래 내역이 없습니다.")
        else:
            for record in self.transaction_history:
                print(record)
        print("----------------------")


#코드 실행
# 첫 번째 계좌 생성
my_account = Account(owner_name="김규빈", balance=1000)
# 입금 및 출금 테스트
my_account.deposit(1000)
my_account.deposit(1000)
my_account.deposit(1000)
my_account.deposit(1000)
my_account.deposit(1000) # 5번째 입금,이자 발생
my_account.deposit(1000)
my_account.deposit(1000)
my_account.deposit(1000)
my_account.deposit(1000)
my_account.deposit(1000) # 10번째 입금, 이자 발생
my_account.withdraw(3000)

# 거래 내역 조회
my_account.history()

# 두 번째 계좌 생성
print("\n" + "="*30 + "\n")
second_account = Account(owner_name="빈규김", balance=50000)
second_account.deposit(10000)
second_account.withdraw(5000)

# 총 계좌 수 확인
Account.get_account_num()

==============================

[김규빈님의 계좌가 개설되었습니다]
은행명: 모두은행
계좌번호: 020-62-211182
초기잔액: 1,000원

1회: 입금	금액: 1,000  	잔액: 2,000
2회: 입금	금액: 1,000  	잔액: 3,000
3회: 입금	금액: 1,000  	잔액: 4,000
4회: 입금	금액: 1,000  	잔액: 5,000
5회: 입금	금액: 1,000  	잔액: 6,000
6회: 이자지급	금액: 60     	잔액: 6,060
7회: 입금	금액: 1,000  	잔액: 7,060
8회: 입금	금액: 1,000  	잔액: 8,060
9회: 입금	금액: 1,000  	잔액: 9,060
10회: 입금	금액: 1,000  	잔액: 10,060
11회: 입금	금액: 1,000  	잔액: 11,060
12회: 이자지급	금액: 110    	잔액: 11,170
13회: 출금	금액: 3,000  	잔액: 8,170

------ 거래 내역 ------
1회: 입금	금액: 1,000  	잔액: 2,000
2회: 입금	금액: 1,000  	잔액: 3,000
3회: 입금	금액: 1,000  	잔액: 4,000
4회: 입금	금액: 1,000  	잔액: 5,000
5회: 입금	금액: 1,000  	잔액: 6,000
6회: 이자지급	금액: 60     	잔액: 6,060
7회: 입금	금액: 1,000  	잔액: 7,060
8회: 입금	금액: 1,000  	잔액: 8,060
9회: 입금	금액: 1,000  	잔액: 9,060
10회: 입금	금액: 1,000  	잔액: 10,060
11회: 입금	금액: 1,000  	잔액: 11,060
12회: 이자지급	금액: 110    	잔액: 11,170
13회: 출금	금액: 3,000  	잔액: 8,170
----------------------

==============================

[빈규김님의 계좌가 개설되었습니다]
은행명: 모두은행
계좌번호: 248-49-093113
초기잔액: 50,000원

1회: 입금	금액: 10,000 	잔액: 60,000
2회: 출금	금액: 5,000  	잔액: 55,000

총 생성된 계좌의 개수: 2개

- [Y]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭을 왜 핵심적이라고 생각하는지 확인
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드의 기능, 존재 이유, 작동 원리 등을 기술했는지 확인
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부

    저는 규빈님의 전체 코드 중에서 제 코드에는 없으면서도, 
    제 코드에 추가로 적용하면 좋을만한 파트를 가져왔습니다.
    
    # class Account - def __init__ 내에 초기 잔액 설정 파트
    if balance < 0:
        raise ValueError("초기 잔액은 0 이상이어야 합니다.")
    self.balance = balance
        
    퍼실님이 업로드해주신 문제 조건에 충실히만 작성했기에, 
    제 코드에는 초기 잔액이 음수가 되지 않도록 방지하는 파트가 없거든요.
    그래서 코드를 쭉 보다가 규빈님이 추가해주신 초기 잔액 설정 및 검증 파트를
    제 코드에도 추가하면 좋을 거 같다고 생각했습니다!

- [Y]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 프로젝트 평가 기준에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부

   그리고 거래횟수를 늘리는 코드가 입금, 출금에도 들어가고 
   그 틀이 되는 부분을 클래스 안에 적어야 하는 부분에서 순서가 헷갈려서 여러차례 오류가 발생했다. 
   클래스를 다루는 일에 익숙하지 않았으나 코드를 역추적해 따라 올라가다보니 출력이 되었다.

   회고에 오류 해결 과정을 작성해주셨는데,
   아마 self.transaction_num += 1 부분 말씀하시는 거 같네요!    

   제 코드에는 _record 함수 호출하면 내역이 쌓이긴 하지만, 
   거래 횟수를 카운팅하고 거래 순번을 추적할 수 있는 코드는 없어서 
   해당 부분 참고할 만한 거 같습니다!!
     
- [Y]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
     
    계좌번호를 생성할때 자릿수에 맞지 않게 나오는 문제가 있었다. 
    예를 들어 랜덤으로 7이라는 숫자가 나왔을때 007이런 계좌 형식이 아닌, 
    7 숫자 하나만 나와서 이 부분을 수정하였다.

    회고에 이렇게 적어주신 부분이 있는데,
    저도 사실 이러한 문제가 똑같이 발생하였었습니다.
    저는 str(random.randint(0, 999)).zfill(3) 로 처리를 했는데,
    규빈님은 f"{random.randint(0, 999):03d}" 이렇게 f-string 포맷을 지정하였네요
    
    제 코드는 숫자 뿐만이 아닌 문자열까지 사용이 가능하지만,
    숫자를 str로 바꿔야 해서 타입 변환이 한 번 더 들어갔는데
    규빈님의 코드는 숫자 포맷 전용으로 리뷰어가 보기에는 의미가 직관적으로 바로 느껴지네요!
   
- [Y]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화/모듈화했는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
   
   # 입금 시 로그
   log_msg = f"{self.transaction_num}회: 입금\t금액: {amount:<7,}\t잔액: {self.balance:,}"
        self.transaction_history.append(log_msg)
        print(log_msg)

   # 5회차 입금마다 이자 지급 로그
   interest_msg = f"{self.transaction_num}회: 이자지급\t금액: {interest:<7,}\t잔액: {self.balance:,}"
            self.transaction_history.append(interest_msg)
            print(interest_msg)

   # 출금 시 로그
   log_msg = f"{self.transaction_num}회: 출금\t금액: {amount:<7,}\t잔액: {self.balance:,}"
        self.transaction_history.append(log_msg)
        print(log_msg)

  저는 규빈님 결과물 출력 부분 포맷이 되게 정갈하고 이쁘게 나오는 걸 보면서
  로그 기록 부분들을 제 코드에도 적용 시켜보고 싶다는 생각이 들었습니다!

# 회고(참고 링크 및 코드 개선)
```
많이 배웠습니다!!
```