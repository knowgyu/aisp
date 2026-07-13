# AISP 데이터 과목. 시험 초점과 실습 지도

이 문서는 데이터 과목(Time Series 01–04, Recommender System 05–08)을 한 번에 복원하기 위한 한국어 오버레이 노트다. PDF와 원본 notebook은 수정하지 않는다.

## 원본과 학습 범위

- 시계열: `data/1-ts-basics.pdf`, `data/2-ts-advanced.pdf`, `data/3_4-ts-practice/ts_practice.ipynb`
- 추천시스템: `data/5-recsys-basics.pdf`, `data/6-recsys-advanced.pdf`, `data/7_8-recsys-practice/RecSys_GCF_practice.ipynb`, `RecSys_NCF.ipynb`
- 데이터: `data/7_8-recsys-practice/ml-latest-small/{ratings,movies,tags,links}.csv`

## 강의자료 바로 열기

- [Time Series 01. 기초 PDF](../../data/1-ts-basics.pdf)
- [Time Series 02. 심화 PDF](../../data/2-ts-advanced.pdf)
- [Time Series 03–04. 실습 PDF](../../data/3_4-ts-practice.pdf)
- [Recommender System 05. 기초 PDF](../../data/5-recsys-basics.pdf)
- [Recommender System 06. 심화 PDF](../../data/6-recsys-advanced.pdf)
- [Recommender System 07–08. 실습 PDF](../../data/7_8-recsys-practice.pdf)

## 시험에서 구분할 두 문제

| 축 | 시계열 예측 | 추천시스템 |
|---|---|---|
| 입력 | 시간 순서가 있는 관측값 | 사용자–아이템 상호작용 |
| 목표 | 다음 시점/다음 구간의 값 | 선호 점수 또는 Top-k 아이템 |
| 핵심 누수 | 미래 데이터를 scaler/학습에 사용 | test interaction을 message passing에 사용 |
| 대표 모델 | LSTM, RNN, 1D CNN, encoder–decoder | NCF, NGCF/GCF |
| 지표 | RMSE, MAPE | RMSE, Precision@k, Recall@k, NDCG@k |

## 코드 복원 체크리스트

1. **전처리**: train 구간으로만 `MinMaxScaler.fit`; test에는 `transform`만 적용한다.
2. **shape**: 시계열 입력은 `(batch, sequence_length, features)`, 추천 입력은 user/item 인덱스 `(batch,)`다.
3. **목표와 출력**: 단일 예측은 `(batch, 1)` 또는 마지막 시점 `(batch,)`; 다중 예측은 `(batch, target_len, 1)`이다.
4. **평가**: scaled 값으로 계산했다면 필요시 `inverse_transform` 후 원 단위 지표를 별도로 확인한다.
5. **그래프 추천**: `edge_index`의 item node index가 user node 수만큼 offset 되었는지 확인한다.

## 복습 순서

```text
PDF 개념 → notebook 전처리 → batch shape 확인 → forward 출력 → loss → metric → 누수/경계 검토
```

## 빠른 암기

- sliding window는 `X[i:i+L] -> y[i+L]`이다.
- LSTM의 `output[:, -1, :]`는 마지막 시간점의 hidden representation이다.
- NCF는 user/movie embedding을 concat해 MLP로 평점값을 회귀한다.
- NGCF는 `edge_index`의 이웃 정보를 전파해 user/item embedding을 갱신한다.
- RMSE는 큰 오차에 민감하고, MAPE는 실제값 0 근처에서 불안정하다.
- Precision@k는 추천한 k개 중 관련 아이템 비율, Recall@k는 관련 아이템 중 회수한 비율이다.

## 원본 보존 규칙

이 파일은 시험 대비 설명만 제공한다. 실행 가능한 코드는 원본 notebook과 정답/실습본을 기준으로 확인하며, PDF·CSV·ipynb를 덮어쓰지 않는다.
