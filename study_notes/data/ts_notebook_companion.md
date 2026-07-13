# 시계열 Notebook Companion: window, RNN/LSTM, 평가

대상 원본: `study_viewer/notebooks/data/ts_basics.ipynb`, `ts_advanced.ipynb`, `ts_practice.ipynb`, `ts_solution.ipynb` 및 `data/1-ts-basics.pdf`, `data/2-ts-advanced.pdf`

## 1. 표에서 supervised sample 만들기

관측값 `y_0,...,y_{T-1}`와 lookback `L`, horizon `H`가 있을 때
`X_i = y[i:i+L]`, `Y_i = y[i+L:i+L+H]`.

- 단변량: `X.shape = [N,L,1]`, `Y.shape = [N,H]`.
- 다변량: `X.shape = [N,L,F]`, `Y.shape = [N,H]`.
- `DataLoader` batch: `[B,L,F]`, target `[B,H]`.
- many-to-one: `out[:, -1, :]` → `[B,H_hidden]` → `Linear(H_hidden,H)` → `[B,H]`.
- many-to-many: sequence output `[B,L,H_hidden]` → head `[B,L,H]`.

## 2. API/객체 역할

| 코드 객체 | 역할 |
|---|---|
| `pd.to_datetime`, `set_index` | 시간축 정렬/인덱스화 |
| `shift(k)` | lag feature 생성 |
| `rolling(window)` | 과거 구간 통계 |
| `MinMaxScaler`/`StandardScaler` | train 기준 scaling |
| custom `Dataset.__getitem__` | window와 target 반환 |
| `nn.RNN`/`nn.LSTM`/`nn.GRU` | sequence representation |
| `nn.Linear` | hidden → 예측 horizon |
| `MSELoss`, `L1Loss` | 예측과 target 비교 |

RNN/LSTM/GRU의 `batch_first=True`를 확인한다. `h_n`과 `c_n`(LSTM)을 sequence output과 혼동하지 않는다.

## 3. 미래 누수와 평가

시간 순서대로 train→validation→test를 나누고 scaler는 train에만 fit한다. rolling/lag는 `shift`와 경계 시점을 확인한다. 평가에서 inverse transform 후 MAE/RMSE를 보고, MAPE는 0 target 처리 규칙을 둔다. baseline(naive last value)보다 개선됐는지 먼저 비교한다.

## 4. 디버깅 순서

```text
index 정렬 -> 결측/주기 확인 -> X/Y 한 샘플 출력
-> batch shape -> model output shape -> loss shape
-> inverse transform -> 예측 plot/MAE RMSE
```

예측이 한 칸 밀리면 `X` 마지막 시점과 `Y` 첫 시점의 경계를 확인한다. `pred [B,1]`와 `target [B]` broadcasting은 허용되더라도 의도하지 않은 loss가 될 수 있으므로 `squeeze`/`reshape`를 명시한다.
