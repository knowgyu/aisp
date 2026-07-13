# Data 과목 시험 포커스: 시계열·추천시스템

이 문서는 `data/1-ts-basics.pdf`부터 `data/7_8-recsys-practice/`까지를 한 번에 복원하는 지도다. 원본 PDF와 노트북은 보존하며, 답안을 외우기보다 **입력 shape → 모델 객체 → loss/metric → 평가 순서**를 설명하는 데 초점을 둔다.

## 1. 출제 포인트 한눈에 보기

| 영역 | 핵심 객체/함수 | 입력·출력 shape | 지표/검사 |
|---|---|---|---|
| 시계열 | window 생성, `Dataset`/`DataLoader`, RNN/LSTM/GRU, `Linear` head | `x=[B,T,F]`, `y=[B,H]` 또는 `[B]`, 예측 `[B,H]` | MAE, MSE, RMSE, 시간 순서 보존 |
| 추천(협업필터링) | user/item id, embedding, GCF/NCF, negative sampling | ids `[B]`, embedding `[B,D]`, score `[B]` | BCE/log loss, RMSE, HR@K, NDCG@K |
| 공통 | train/validation/test 분리, seed, `model.train/eval`, optimizer | batch dimension을 첫 축으로 유지 | leakage, baseline, 과적합 |

## 2. 답안 복원 순서

1. 원본 데이터의 시간축 또는 user-item 행을 확인한다.
2. 전처리 결과가 모델 입력으로 바뀌는 지점을 찾는다 (`__getitem__`, `collate`, embedding lookup).
3. forward에서 각 tensor가 어떻게 결합되는지 shape를 적는다.
4. loss가 예측값과 target의 어떤 축을 비교하는지 확인한다.
5. `optimizer.zero_grad → loss.backward → optimizer.step` 순서를 설명한다.
6. 평가 시 shuffle·dropout·gradient가 꺼졌는지와 지표 방향(낮을수록/높을수록)을 말한다.

## 3. 자주 틀리는 경계

- 시계열 validation에 미래 관측값을 섞으면 leakage다. 랜덤 split 대신 시간 순 split을 우선한다.
- `Embedding`에는 실수 feature가 아니라 정수 index를 넣는다. user/item id를 0부터 연속 index로 매핑한다.
- RMSE/MAE는 낮을수록 좋고, HR@K/NDCG@K는 높을수록 좋다.
- 추천 score가 `[B]`인지 `[B,1]`인지에 따라 `squeeze(-1)`과 target shape를 맞춘다.
- `BCEWithLogitsLoss`에는 sigmoid 이전 logits를 넣고, `BCELoss`에는 확률을 넣는다.

## 4. 최소 회고 체크리스트

- `[B,T,F]`가 왜 `[T,B,F]`로 바뀌는지 (`batch_first`) 설명할 수 있는가?
- one-step 예측과 multi-step 예측의 target shape가 어떻게 다른가?
- user/item embedding 두 벡터를 dot product로 점수화하는 이유는 무엇인가?
- pointwise negative sampling과 pairwise ranking의 차이는 무엇인가?
- 테스트 지표가 좋아도 시간 leakage 또는 인기상품 편향이 없는지 어떻게 점검할 것인가?

원본: `data/1-ts-basics.pdf`, `data/2-ts-advanced.pdf`, `data/3_4-ts-practice.pdf`, `data/5-recsys-basics.pdf`, `data/6-recsys-advanced.pdf`, `data/7_8-recsys-practice/`.
