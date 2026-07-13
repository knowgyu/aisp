# Data 과목 시험 포커스: Time Series와 Recommender System

이 문서는 `data/` PDF 교안과 실습 notebook을 읽기 전에 보는 시험용 지도다. 원본은 수정하지 않고 핵심 객체, shape, 학습 순서와 평가식을 한 곳에 모았다.

## 1. 범위와 복습 순서

| 영역 | 원본 | 우선순위 |
|---|---|---:|
| 시계열 기초·심화 | `data/pdf_markdown/1-ts-basics.md`, `2-ts-advanced.md` | 1 |
| 시계열 실습 | `data/3_4-ts-practice/ts_practice.ipynb`, `ts_solution.ipynb` | 1 |
| 추천 기초·심화 | `data/pdf_markdown/5-recsys-basics.md`, `6-recsys-advanced.md` | 2 |
| 추천 실습 | `data/7_8-recsys-practice/RecSys_GCF_practice.ipynb`, `RecSys_NCF.ipynb` | 2 |

## 2. 시험 답안 공통 프레임

1. **입력/목표**를 먼저 쓴다: 시계열은 과거 window→미래 값, 추천은 user-item interaction→미관측 item 점수.
2. **shape**를 쓴다: 배치 차원, sequence 차원, feature/embedding 차원을 빠뜨리지 않는다.
3. **data leakage**를 확인한다: scaler는 train에만 `fit`, test에는 `transform`만 한다. interaction split 뒤 test edge가 train에 섞이지 않아야 한다.
4. **모델 출력과 loss/metric**을 연결한다: 회귀는 `[B, 1]`과 MSE, ranking은 top-k와 Recall/Precision 계열이다.
5. `train()`/`eval()`, `no_grad()`, 역정규화 여부를 구분한다.

## 3. 비교표

| 질문 | Time Series | Recommender |
|---|---|---|
| 데이터 | 시간 순서가 있는 관측값 | user-item edge/평점 |
| 대표 입력 | `X: [B, L, F]` | `edge_index: [2, E]`, embedding |
| 출력 | 다음 값 또는 horizon | user-item relevance score |
| 학습 | MSE + Adam | BPR/negative sampling 또는 ranking loss |
| 평가 | RMSE, MAPE | Recall@K, Precision@K, NDCG@K |
| 핵심 위험 | 미래를 미리 사용, scaling 오류 | train/test edge leakage, 인기 편향 |

## 4. 반드시 설명할 용어

- **window/lag**: 과거 `L`개 시점을 입력으로 삼는 것.
- **recursive forecast**: 한 번 예측한 값을 다음 입력에 다시 넣어 여러 시점을 예측하는 방식.
- **explicit→implicit**: 평점처럼 수치인 interaction을 threshold로 positive edge로 바꾸는 방식.
- **message passing**: 그래프 이웃 embedding을 선형변환·결합해 user/item 표현을 갱신하는 과정.
- **candidate/ranking**: 전체 item 점수화 후 이미 본 item을 제외하고 top-k를 고르는 과정.

## 5. 출제 전 체크리스트

- [ ] train/test 시간 순서를 보존했는가?
- [ ] `MinMaxScaler.fit`이 train에만 호출되는가?
- [ ] LSTM/CNN/RNN의 입력 차원 변환을 설명할 수 있는가?
- [ ] `edge_index[0]`이 user, `edge_index[1]`이 item이라는 offset 규칙을 확인했는가?
- [ ] 추천 metric에서 이미 관측한 item을 제거했는가?
- [ ] RMSE는 원래 단위로 보고할지 scaled 단위로 보고할지 밝혔는가?
