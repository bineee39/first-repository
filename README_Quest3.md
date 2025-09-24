# AIFFEL Campus Online Code Peer Review Templete
- �ڴ� : ��Ժ��
- ����� : ������

# PRT(Peer Review Template)
- [Y]  **1. �־��� ������ �ذ��ϴ� �ϼ��� �ڵ尡 ����Ǿ�����?**
    - �������� �䱸�ϴ� ���� ������� ÷�εǾ����� Ȯ��
        - �߿�! �ش� ������ �����ϴ� �κ��� ĸ���� �ٰŷ� ÷��
    
import random

class Account:
    account_count = 0
    #account_count: ���ݱ��� ������ �� ���� ���� ������
    def __init__(self, owner_name, balance):
        #owner_name (str): ���� ������ �̸�.
        #balance (int): �ʱ� �ܾ�.
        # ���� �⺻ ���� ����(�����̸��� ����������� ����)
        self.bank_name = "�������"
        self.owner_name = owner_name
        #�ʱ� �ܾ� ����
        if balance < 0:
            raise ValueError("�ʱ� �ܾ��� 0 �̻��̾�� �մϴ�.")
        self.balance = balance

        # 2 ���¹�ȣ ���� ���� (3�ڸ�-2�ڸ�-6�ڸ�)
        num1 = f"{random.randint(0, 999):03d}"
        num2 = f"{random.randint(0, 99):02d}"
        num3 = f"{random.randint(0, 999999):06d}"
        self.account_number = f"{num1}-{num2}-{num3}"

        # 3 ���� ���� ����(��ü ���鶧����+1)
        Account.account_count += 1

        # 4 �ŷ� ���� ���� �ʱ�ȭ
        self.transaction_history = []
        self.deposit_count = 0
        self.transaction_num = 0

        print(f"[{self.owner_name}���� ���°� �����Ǿ����ϴ�]")
        print(f"�����: {self.bank_name}")
        print(f"���¹�ȣ: {self.account_number}")
        print(f"�ʱ��ܾ�: {self.balance:,}��\n")


    def deposit(self, amount):
        #�Ա��Լ�
        if amount < 1:
            print("�Ա��� 1�� �̻� �����մϴ�.")
            return
        #�Ա�Ƚ��,�ŷ�Ƚ�� üũ
        self.balance += amount
        self.deposit_count += 1
        self.transaction_num += 1

        log_msg = f"{self.transaction_num}ȸ: �Ա�\t�ݾ�: {amount:<7,}\t�ܾ�: {self.balance:,}"
        self.transaction_history.append(log_msg)
        print(log_msg)

        # 5ȸ �Աݸ��� 1% ���� ����
        if self.deposit_count % 5 == 0:
            interest = int(self.balance * 0.01)
            self.balance += interest
            self.transaction_num += 1

            interest_msg = f"{self.transaction_num}ȸ: ��������\t�ݾ�: {interest:<7,}\t�ܾ�: {self.balance:,}"
            self.transaction_history.append(interest_msg)
            print(interest_msg)


    def withdraw(self, amount):
        #��� �Լ�
        if amount > self.balance:
            print("����: �ܾ��� �����մϴ�.")
            return
        #�ܰ�-��ݾ�,�ŷ�Ƚ��+1
        self.balance -= amount
        self.transaction_num += 1

        log_msg = f"{self.transaction_num}ȸ: ���\t�ݾ�: {amount:<7,}\t�ܾ�: {self.balance:,}"
        self.transaction_history.append(log_msg)
        print(log_msg)

    #class�� �ϴ� ��
    @classmethod
    def get_account_num(cls):
        #������ �� ���� ���� ���
        print(f"\n�� ������ ������ ����: {cls.account_count}��")


    def history(self):
        #������ ��.��� ������ ��� ���
        print("\n------ �ŷ� ���� ------")
        if not self.transaction_history:
            print("�ŷ� ������ �����ϴ�.")
        else:
            for record in self.transaction_history:
                print(record)
        print("----------------------")


#�ڵ� ����
# ù ��° ���� ����
my_account = Account(owner_name="��Ժ�", balance=1000)
# �Ա� �� ��� �׽�Ʈ
my_account.deposit(1000)
my_account.deposit(1000)
my_account.deposit(1000)
my_account.deposit(1000)
my_account.deposit(1000) # 5��° �Ա�,���� �߻�
my_account.deposit(1000)
my_account.deposit(1000)
my_account.deposit(1000)
my_account.deposit(1000)
my_account.deposit(1000) # 10��° �Ա�, ���� �߻�
my_account.withdraw(3000)

# �ŷ� ���� ��ȸ
my_account.history()

# �� ��° ���� ����
print("\n" + "="*30 + "\n")
second_account = Account(owner_name="��Ա�", balance=50000)
second_account.deposit(10000)
second_account.withdraw(5000)

# �� ���� �� Ȯ��
Account.get_account_num()

==============================

[��Ժ���� ���°� �����Ǿ����ϴ�]
�����: �������
���¹�ȣ: 020-62-211182
�ʱ��ܾ�: 1,000��

1ȸ: �Ա�	�ݾ�: 1,000  	�ܾ�: 2,000
2ȸ: �Ա�	�ݾ�: 1,000  	�ܾ�: 3,000
3ȸ: �Ա�	�ݾ�: 1,000  	�ܾ�: 4,000
4ȸ: �Ա�	�ݾ�: 1,000  	�ܾ�: 5,000
5ȸ: �Ա�	�ݾ�: 1,000  	�ܾ�: 6,000
6ȸ: ��������	�ݾ�: 60     	�ܾ�: 6,060
7ȸ: �Ա�	�ݾ�: 1,000  	�ܾ�: 7,060
8ȸ: �Ա�	�ݾ�: 1,000  	�ܾ�: 8,060
9ȸ: �Ա�	�ݾ�: 1,000  	�ܾ�: 9,060
10ȸ: �Ա�	�ݾ�: 1,000  	�ܾ�: 10,060
11ȸ: �Ա�	�ݾ�: 1,000  	�ܾ�: 11,060
12ȸ: ��������	�ݾ�: 110    	�ܾ�: 11,170
13ȸ: ���	�ݾ�: 3,000  	�ܾ�: 8,170

------ �ŷ� ���� ------
1ȸ: �Ա�	�ݾ�: 1,000  	�ܾ�: 2,000
2ȸ: �Ա�	�ݾ�: 1,000  	�ܾ�: 3,000
3ȸ: �Ա�	�ݾ�: 1,000  	�ܾ�: 4,000
4ȸ: �Ա�	�ݾ�: 1,000  	�ܾ�: 5,000
5ȸ: �Ա�	�ݾ�: 1,000  	�ܾ�: 6,000
6ȸ: ��������	�ݾ�: 60     	�ܾ�: 6,060
7ȸ: �Ա�	�ݾ�: 1,000  	�ܾ�: 7,060
8ȸ: �Ա�	�ݾ�: 1,000  	�ܾ�: 8,060
9ȸ: �Ա�	�ݾ�: 1,000  	�ܾ�: 9,060
10ȸ: �Ա�	�ݾ�: 1,000  	�ܾ�: 10,060
11ȸ: �Ա�	�ݾ�: 1,000  	�ܾ�: 11,060
12ȸ: ��������	�ݾ�: 110    	�ܾ�: 11,170
13ȸ: ���	�ݾ�: 3,000  	�ܾ�: 8,170
----------------------

==============================

[��Ա���� ���°� �����Ǿ����ϴ�]
�����: �������
���¹�ȣ: 248-49-093113
�ʱ��ܾ�: 50,000��

1ȸ: �Ա�	�ݾ�: 10,000 	�ܾ�: 60,000
2ȸ: ���	�ݾ�: 5,000  	�ܾ�: 55,000

�� ������ ������ ����: 2��

- [Y]  **2. ��ü �ڵ忡�� ���� �ٽ����̰ų� ���� �����ϰ� �����ϱ� ����� �κп� �ۼ��� 
�ּ� �Ǵ� doc string�� ���� �ش� �ڵ尡 �� ���صǾ�����?**
    - �ش� �ڵ� ���� �� �ٽ����̶�� �����ϴ��� Ȯ��
    - �ش� �ڵ� ���� doc string/annotation�� �޷� �ִ��� Ȯ��
    - �ش� �ڵ��� ���, ���� ����, �۵� ���� ���� ����ߴ��� Ȯ��
    - �ּ��� ���� �ڵ� ���ذ� �� �Ǿ����� Ȯ��
        - �߿�! �� �ۼ��Ǿ��ٰ� �����Ǵ� �κ��� ĸ���� �ٰŷ� ÷��

    ���� �Ժ���� ��ü �ڵ� �߿��� �� �ڵ忡�� �����鼭��, 
    �� �ڵ忡 �߰��� �����ϸ� �������� ��Ʈ�� �����Խ��ϴ�.
    
    # class Account - def __init__ ���� �ʱ� �ܾ� ���� ��Ʈ
    if balance < 0:
        raise ValueError("�ʱ� �ܾ��� 0 �̻��̾�� �մϴ�.")
    self.balance = balance
        
    �۽Ǵ��� ���ε����ֽ� ���� ���ǿ� ������� �ۼ��߱⿡, 
    �� �ڵ忡�� �ʱ� �ܾ��� ������ ���� �ʵ��� �����ϴ� ��Ʈ�� ���ŵ��.
    �׷��� �ڵ带 �� ���ٰ� �Ժ���� �߰����ֽ� �ʱ� �ܾ� ���� �� ���� ��Ʈ��
    �� �ڵ忡�� �߰��ϸ� ���� �� ���ٰ� �����߽��ϴ�!

- [Y]  **3. ������ �� �κ��� ������Ͽ� ������ �ذ��� ����� ����ų�
���ο� �õ� �Ǵ� �߰� ������ �����غó���?**
    - ���� ���� �� �ذ� ������ �� ����Ͽ����� Ȯ��
    - ������Ʈ �� ���ؿ� ���� �߰������� ������ ������ �õ�, 
    ������ ��ϵǾ� �ִ��� Ȯ��
        - �߿�! �� �ۼ��Ǿ��ٰ� �����Ǵ� �κ��� ĸ���� �ٰŷ� ÷��

   �׸��� �ŷ�Ƚ���� �ø��� �ڵ尡 �Ա�, ��ݿ��� ���� 
   �� Ʋ�� �Ǵ� �κ��� Ŭ���� �ȿ� ����� �ϴ� �κп��� ������ �򰥷��� �������� ������ �߻��ߴ�. 
   Ŭ������ �ٷ�� �Ͽ� �ͼ����� �ʾ����� �ڵ带 �������� ���� �ö󰡴ٺ��� ����� �Ǿ���.

   ȸ�� ���� �ذ� ������ �ۼ����̴ּµ�,
   �Ƹ� self.transaction_num += 1 �κ� �����Ͻô� �� ���׿�!    

   �� �ڵ忡�� _record �Լ� ȣ���ϸ� ������ ���̱� ������, 
   �ŷ� Ƚ���� ī�����ϰ� �ŷ� ������ ������ �� �ִ� �ڵ�� ��� 
   �ش� �κ� ������ ���� �� �����ϴ�!!
     
- [Y]  **4. ȸ�� �� �ۼ��߳���?**
    - �־��� ������ �ذ��ϴ� �ϼ��� �ڵ� ���� ������Ʈ ������� ����
    ������� �ƽ�����, ������ ���� ��ϵǾ� �ִ��� Ȯ��
    - ��ü �ڵ� ���� �÷ο츦 �׷����� �׷��� ���ظ� ���� �ִ��� Ȯ��
        - �߿�! �� �ۼ��Ǿ��ٰ� �����Ǵ� �κ��� ĸ���� �ٰŷ� ÷��
     
    ���¹�ȣ�� �����Ҷ� �ڸ����� ���� �ʰ� ������ ������ �־���. 
    ���� ��� �������� 7�̶�� ���ڰ� �������� 007�̷� ���� ������ �ƴ�, 
    7 ���� �ϳ��� ���ͼ� �� �κ��� �����Ͽ���.

    ȸ�� �̷��� �����ֽ� �κ��� �ִµ�,
    ���� ��� �̷��� ������ �Ȱ��� �߻��Ͽ������ϴ�.
    ���� str(random.randint(0, 999)).zfill(3) �� ó���� �ߴµ�,
    �Ժ���� f"{random.randint(0, 999):03d}" �̷��� f-string ������ �����Ͽ��׿�
    
    �� �ڵ�� ���� �Ӹ��� �ƴ� ���ڿ����� ����� ����������,
    ���ڸ� str�� �ٲ�� �ؼ� Ÿ�� ��ȯ�� �� �� �� ���µ�
    �Ժ���� �ڵ�� ���� ���� �������� ���� ���⿡�� �ǹ̰� ���������� �ٷ� �������׿�!
   
- [Y]  **5. �ڵ尡 �����ϰ� ȿ�����ΰ���?**
    - ���̽� ��Ÿ�� ���̵� (PEP8) �� �ؼ��Ͽ����� Ȯ��
    - �ڵ� �ߺ��� �ּ�ȭ�ϰ� ���������� ����� �� �ֵ��� �Լ�ȭ/���ȭ�ߴ��� Ȯ��
        - �߿�! �� �ۼ��Ǿ��ٰ� �����Ǵ� �κ��� ĸ���� �ٰŷ� ÷��
   
   # �Ա� �� �α�
   log_msg = f"{self.transaction_num}ȸ: �Ա�\t�ݾ�: {amount:<7,}\t�ܾ�: {self.balance:,}"
        self.transaction_history.append(log_msg)
        print(log_msg)

   # 5ȸ�� �Աݸ��� ���� ���� �α�
   interest_msg = f"{self.transaction_num}ȸ: ��������\t�ݾ�: {interest:<7,}\t�ܾ�: {self.balance:,}"
            self.transaction_history.append(interest_msg)
            print(interest_msg)

   # ��� �� �α�
   log_msg = f"{self.transaction_num}ȸ: ���\t�ݾ�: {amount:<7,}\t�ܾ�: {self.balance:,}"
        self.transaction_history.append(log_msg)
        print(log_msg)

  ���� �Ժ�� ����� ��� �κ� ������ �ǰ� �����ϰ� �̻ڰ� ������ �� ���鼭
  �α� ��� �κе��� �� �ڵ忡�� ���� ���Ѻ��� �ʹٴ� ������ ������ϴ�!

# ȸ��(���� ��ũ �� �ڵ� ����)
```
���� ������ϴ�!!
```