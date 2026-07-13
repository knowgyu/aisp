# Data 과목 시험 초점: 시계열·추천시스템 한 장 지도

대상 원본: `data/1-ts-basics.pdf`, `data/2-ts-advanced.pdf`, `data/5-recsys-basics.pdf`, `data/6-recsys-advanced.pdf`, `data/7_8-recsys-practice/`

## 1. 출제 흐름

```text
원본 표/로그 -> pandas 전처리 -> (시계열) window/lag 또는 (추천) user-item 상호작용
-> tensor shape 확인 -> model forward -> loss -> metric/시각화
```

| 영역 | 입력 단위 | 핵심 모델/객체 | 대표 metric | 시험에서 확인할 것 |
|---|---|---|---|---|
| 시계열 | 시간 순서가 있는 `y_t`, 외생변수 `x_t` | lag, rolling, RNN/LSTM/GRU, forecasting head | MAE, MSE/RMSE, MAPE | 시간 순서 보존, look-ahead 누출, window shape |
| 추천 | `(user_id, item_id, rating)` 또는 implicit click | embedding, MF/GCF, NCF/MLP | RMSE/MAE, Precision@K, Recall@K, NDCG@K | ID mapping, negative sampling, score shape |

## 2. shape 빠른 복원

- 시계열 단변량 batch: `x.shape == [B, L, F]` (`B`: batch, `L`: lookback, `F`: feature 수).
- PyTorch `batch_first=True` RNN: 입력 `[B,L,F]`, 출력 sequence `[B,L,H]`, hidden `[num_layers*num_directions,B,H]`.
- 마지막 시점 예측: `out[:, -1, :] -> [B,H] -> Linear(H, horizon) -> [B,horizon]`.
- 추천 mini-batch: user/item ID `[B]`; embedding lookup 결과 `[B,D]`; dot product `[B]`; NCF concat `[B,2D]` 후 MLP `[B,1]`.
- GCF graph propagation: node embedding `[N,D]`가 adjacency 정규화로 `[N,D]`를 반복 변환하고, user/item 행을 선택해 pair score `[B]`를 만든다.

## 3. 데이터 누수 체크

1. train/validation/test를 시간 기준으로 나눈다(랜덤 shuffle 금지).
2. rolling 통계는 현재 시점 이전 값만 사용한다.
3. scaler는 train 구간에만 `fit`, validation/test에는 `transform`만 한다.
4. 추천 평가에서 train interaction을 후보에서 제외한다.
5. `item_id`/`user_id`를 문자열 그대로 모델에 넣지 말고 contiguous index로 mapping한다.

## 4. metric 선택

- 회귀 예측: `MAE = mean(|y-ŷ|)`, `MSE = mean((y-ŷ)^2)`, `RMSE = sqrt(MSE)`. 큰 오차를 강하게 벌주면 RMSE, 해석 가능한 평균 절대 오차는 MAE.
- MAPE는 `y=0`에서 불안정하므로 0 처리 규칙을 명시한다.
- Top-K 추천: `Precision@K = hits/K`, `Recall@K = hits/|relevant|`, `NDCG@K`는 순위가 앞일수록 큰 gain을 준다.
- rating prediction과 ranking recommendation은 목표가 다르므로 RMSE만으로 top-K 품질을 주장하지 않는다.

## 5. 시험 답안 템플릿

> 입력의 각 축은 무엇인가? (`[B,L,F]` 또는 user/item ID `[B]`) → 어떤 객체가 shape를 바꾸는가? (window, embedding, transpose, `unsqueeze`) → loss가 비교하는 두 tensor shape는 같은가? → 평가 시 미래 정보/seen item이 섞이지 않았는가?

원본 PDF와 notebook은 변경하지 않았으며 이 파일은 viewer용 overlay다.

## 6. Notebook API 앵커

- 시계열 notebook은 `yfinance.download("GOOG", start="2020-01-01", end="2024-12-31")`로 `Open`을 가져오고, `MinMaxScaler`, `DataLoader`, `nn.LSTM`, `MSELoss`, `Adam` API를 연결한다.
- 추천 notebook은 MovieLens `ratings.csv`를 읽고 `LabelEncoder`, `nn.Embedding`, `torch.topk`와 RMSE/Precision@10/Recall@10/NDCG@10을 사용한다.
