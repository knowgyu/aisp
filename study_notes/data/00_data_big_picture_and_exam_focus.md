# Data 큰 그림과 시험 포커스

이 문서는 Data 과목의 **Time Series(시계열)**와 **Recommender System(추천 시스템)**을 원본 PDF와 실습 notebook에 들어가기 전에 연결해 보는 시험 대비 overlay다. 원본 자료와 notebook은 수정하지 않는다.

## 1. 원본 자료와 권장 순서

| 순서 | 주제 | 원본 PDF/노트 | 실습 notebook |
|---:|---|---|---|
| 1 | 시계열 기초 | [`1-ts-basics.pdf`](../../study_viewer/data/1-ts-basics.pdf) / [`ts_basics.md`](ts_basics.md) | — |
| 2 | 시계열 심화 | [`2-ts-advanced.pdf`](../../study_viewer/data/2-ts-advanced.pdf) / [`ts_advanced.md`](ts_advanced.md) | — |
| 3 | 시계열 실습 | [`3_4-ts-practice.pdf`](../../study_viewer/data/3_4-ts-practice.pdf) / [`ts_practice_material.md`](ts_practice_material.md) | `data/3_4-ts-practice/ts_practice.ipynb`, `ts_solution.ipynb` |
| 4 | 추천 기초 | [`5-recsys-basics.pdf`](../../study_viewer/data/5-recsys-basics.pdf) / [`recsys_basics.md`](recsys_basics.md) | — |
| 5 | 추천 심화 | [`6-recsys-advanced.pdf`](../../study_viewer/data/6-recsys-advanced.pdf) / [`recsys_advanced.md`](recsys_advanced.md) | — |
| 6 | 추천 실습 | [`7_8-recsys-practice.pdf`](../../study_viewer/data/7_8-recsys-practice.pdf) / [`recsys_practice_material.md`](recsys_practice_material.md) | `data/7_8-recsys-practice/RecSys_GCF_practice.ipynb`, `RecSys_NCF.ipynb` |

## 2. 두 영역을 하나의 pipeline으로 보기

```text
Time Series: 관측값 -> 시간 순서 split -> scaling -> window [B,L,F]
            -> LSTM/CNN/RNN -> 미래값 -> RMSE/MAPE

Recommender: user-item interaction -> implicit edge -> edge_index [2,E]
             -> embedding/message passing -> score(u,i) -> top-K -> Recall/NDCG
```

| 시험 질문 | Time Series | Recommender System |
|---|---|---|
| 입력 | 과거 `L`개 시점의 feature | user-item interaction graph |
| 대표 shape | `X: [B, L, F]`, `y: [B, 1]` | `edge_index: [2, E]`, user `[U,D]`, item `[I,D]` |
| 출력 | one-step `[B,1]` 또는 horizon `[B,H,1]` | relevance score / top-K item |
| 학습 | `MSELoss` + Adam | graph/MLP representation + ranking objective |
| 평가 | RMSE, MAPE | Recall@K, Precision@K, NDCG@K |
| 대표 누수 | test로 scaler를 fit, 미래 window 사용 | test edge를 message passing에 사용, seen item 미제거 |

## 3. 출제포인트 매핑: 코드에서 설명할 것

### 3.1 Time Series

1. **분할과 scaling**: 시간 순서를 지켜 train/test를 나누고 `MinMaxScaler.fit_transform(train)` 뒤 `transform(test)`만 한다.
2. **Sliding window**: `sequence_length=50`이면 `X[i] = data[i:i+50]`, `y[i] = data[i+50]`이므로 `X`는 `[N,50,1]`이다.
3. **모델 역할**: LSTM/RNN은 sequence의 hidden state를 만들고 마지막 출력 `[B,HIDDEN]`을 `Linear(HIDDEN,1)`에 보낸다. Conv1D는 channel-first `[B,F,L]`가 필요하다.
4. **평가 모드**: `model.eval()`과 `torch.no_grad()`로 평가하고, scaled 예측을 원래 단위로 비교할지 명시한다.
5. **multi-step**: recursive forecast는 직전 예측을 다음 window에 다시 넣는다. encoder-decoder는 입력 `[B,50,1]`에서 여러 시점 `[B,10,1]`을 만든다.

### 3.2 Recommender System

1. **explicit→implicit**: 평점 또는 interaction을 threshold 기준 positive edge로 만든다.
2. **node id와 edge**: `edge_index[0]`은 user, `edge_index[1]`은 item이다. user/item id를 하나의 graph node 공간에 넣으면 item에 `num_users` offset을 둘 수 있다.
3. **message passing**: 이웃 embedding을 aggregate하고 선형변환/activation/dropout으로 user/item 표현을 갱신한다. adjacency matrix 대신 sparse `edge_index: [2,E]`를 사용한다.
4. **score와 ranking**: user·item embedding 내적 또는 NCF MLP 출력으로 score를 만들고, 이미 본 train item을 제외한 뒤 top-K를 평가한다.
5. **평가 의미**: RMSE는 수치 예측, Recall/Precision/NDCG는 순위 품질이다. 추천 문제를 RMSE 하나로 설명하지 않는다.

## 4. 시험 직전 shape·API 체크표

| 확인할 코드 | 기대 내용 |
|---|---|
| `train_loader` | `TensorDataset(X, y)`, batch 예: `32` |
| `nn.LSTM(..., batch_first=True)` | 입력 `[B,L,F]`, 출력 `[B,L,H]` |
| `nn.Conv1d` | 입력을 `[B,F,L]`로 transpose |
| `MSELoss` | 예측과 target의 batch/출력 shape 일치 |
| `create_edge_index` | threshold 적용, user/item index 및 offset 보존 |
| `NGCFLayer` | 이웃 aggregate 후 `W1`, `W2` 변환 |
| `evaluate(..., k)` | test positive와 top-k candidate 비교, seen item 제거 |

## 5. 한 문장 답안 템플릿

- 시계열: “과거 `L`개 관측치를 `[B,L,F]`로 windowing하고 train에만 scaler를 fit한 뒤, sequence model의 마지막 hidden output으로 다음 값을 회귀하며 RMSE/MAPE로 평가한다.”
- 추천: “user-item interaction을 `edge_index [2,E]`로 구성하고 graph/embedding propagation으로 user·item 표현을 만든 뒤, score 상위 K개를 추천하며 seen item을 제외한 Recall/NDCG로 평가한다.”

## 6. 최종 체크리스트

- [ ] PDF 개념과 notebook의 실제 변수/API를 연결했는가?
- [ ] 입력, 중간, 출력 shape를 최소 한 번 적었는가?
- [ ] train/test 경계와 leakage 방지 방법을 설명했는가?
- [ ] `train()`/`eval()` 및 `no_grad()`의 차이를 아는가?
- [ ] 회귀 metric과 ranking metric을 구분했는가?
- [ ] 원본 PDF와 notebook은 보존되고 overlay만 추가되었는가?
