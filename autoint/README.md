# 결과물 사진

<img width="1087" height="754" alt="image" src="https://github.com/user-attachments/assets/df35b2f0-5bf0-4f2d-8146-6f488fb14a6b" /><img width="1164" height="583" alt="image" src="https://github.com/user-attachments/assets/e760c8e3-e3c4-4ea9-9806-7293dd94ee81" />

<img width="1087" height="754" alt="image" src="https://github.com/user-attachments/assets/81d156f7-e1c6-45a2-9297-9cbd12bc25ce" />

<img width="1093" height="626" alt="image" src="https://github.com/user-attachments/assets/cd83a60a-bc1f-4664-ac59-94fb01be8379" />

# 하이퍼파라미터 조절 

1. 원본
   
<img width="1570" height="327" alt="image" src="https://github.com/user-attachments/assets/296d69c4-08c5-4c1f-9b56-771299fecdf8" />


2. 조정


하이퍼파라미터 수정

학습률 0.0001 -> 0.001 ,학습 속도를 10배 높여 더 빠르게 수렴하도록 조정했습니다.
배치 사이즈 2048 ->256 배치 크기를 줄여 가중치 업데이트 빈도를 높였습니다.
드롭아웃0.4-> 0.2 정보를 덜 소실시키기 위해 드롭아웃 비율을 낮췄습니다.
임베딩 크기 유지했습니다.
에포크는 5로 유지했습니다.  

<img width="1138" height="222" alt="image" src="https://github.com/user-attachments/assets/ee6b09d7-ad52-42db-aee6-cf43beb9e910" />  

3. 결과
Loss 감소: Validation Loss가 기존 0.5414에서 0.5225로 약 3.5% 감소하여 예측 오차를 줄이는 데 성공했습니다.

학습 효율 증대: 수정된 모델은 1 Epoch 만에 기존 모델의 최종 성능(5 Epoch)에 근접한 Loss(0.5421)를 기록하엿습니다. 이는 MLP 결합과 학습률 조정이 초기 학습 효율을 크게 높였음을 시사합니다.

최종 정확도: 결과적으로 73.71% (val_accuracy)의 검증 정확도를 달성하며 유의미한 성능 향상을 확인하였습니다.

