# Time Series 03–04. LSTM·CNN·RNN notebook 동반 복습

대상 원본: `data/1-ts-basics.pdf`, `data/2-ts-advanced.pdf`, `data/3_4-ts-practice/ts_practice.ipynb`. 목표는 GOOG Open 가격 예측 notebook의 실행 순서와 tensor shape를 복원하는 것이다.

## 전처리 흐름

```text
yf.download('GOOG') → Open column → 시간순 80/20 split
→ MinMaxScaler(train fit) → sliding window(lookback=50)
→ TensorDataset/DataLoader → model → inverse scale/metric
```

| 코드 객체 | 역할 | shape/주의 |
|---|---|---|
| `df` | OHLCV DataFrame | 행=날짜, 선택 feature `Open` |
| `train_data`, `test_data` | 시간순 분할 | 예시 `(1006,1)`, `(251,1)` |
| `scaler` | `[0,1]` 정규화 | train에만 `fit` |
| `X_train`, `X_test` | 50일 window | `(N,50,1)` |
| `y_train`, `y_test` | 다음 시점 target | `(N,)` 또는 `(N,1)` |
| `DataLoader` | batch 공급 | `batch_x=(B,50,1)` |

Sliding window는 `X[i:i+50] -> y[i+50,0]`이다. `sequence_length`를 바꾸면 샘플 수 `len(data)-sequence_length`도 함께 바뀐다.

## 모델 forward와 shape

- LSTM: `nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)`의 output은 `(B,50,64)`; 마지막 시점에 `Linear(64,1)`을 적용해 `(B,50,1)`을 얻고 `[:, -1, 0]`을 단일 예측으로 쓴다.
- RNN: `nn.RNN(1,64,2,batch_first=True)`도 동일한 입력 shape를 사용한다.
- CNN: 입력 `(B,50,1)`을 Conv1d용 `(B,1,50)`으로 transpose한다. convolution 후 다시 sequence 축을 맞추고 마지막/flatten shape가 `Linear` 입력과 일치해야 한다.
- Encoder–decoder: encoder 입력 `(B,50,1)`, decoder target은 `(B,10,1)`(`target_len=10`), 출력도 `(B,10,1)`이다.

## 학습·평가 API

```python
loss_fn = nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train(); loss.backward(); optimizer.step()
model.eval(); with torch.no_grad(): ...
```

`root_mean_squared_error(y_true, y_pred)`와 `mean_absolute_percentage_error(y_true, y_pred)`는 같은 단위/scale에서 비교한다. scaled metric과 원 가격 metric을 섞지 않는다. MAPE는 실제값 0에 취약하다.

## 예측 시 주의점

`plot_forecasting`의 rolling forecast는 직전 예측을 다음 입력 window에 넣으므로 시간이 길어질수록 오차가 누적된다. `input_data`가 `(50,)`이면 model 입력을 `.view(1,-1,1)`로 바꾼다. GPU tensor는 `.cpu().numpy()` 전에 CPU로 이동한다.

## 시험 직전 체크

- scaler를 전체 df에 fit하면 미래 정보 누수라는 점을 설명한다.
- `batch_first=True`가 `(B,T,F)` 순서를 의미함을 안다.
- 단일-step과 multi-step의 target shape 차이를 말할 수 있다.
- RMSE/MAPE 호출 인자 순서가 `y_true, y_pred`인지 확인한다.
- 현재 notebook의 TODO 빈칸은 학습용이므로 원본을 수정하지 않고 정답/실습본에서 채운다.
