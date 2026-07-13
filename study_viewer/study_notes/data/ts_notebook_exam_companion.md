# Time Series Notebook 시험 동반 노트: LSTM·CNN·RNN

원본: `data/3_4-ts-practice/ts_practice.ipynb`, 해설: `data/3_4-ts-practice/ts_solution.ipynb`.

## 1. 전처리와 shape

```text
Yahoo Finance Open -> train/test (80/20)
train fit MinMaxScaler -> sliding window L=50
X: [N, 50, 1], y: [N, 1] -> TensorDataset -> DataLoader(batch=32)
```

`train_data`와 `test_data`는 각각 `(1006, 1)`, `(251, 1)` 예시다. `scaler.fit_transform(train)` 후 `scaler.transform(test)`만 호출해야 미래 통계 누수를 막는다. window 함수는 `features[i:i+50]`, `label[i+50,0]`을 만든다.

## 2. 모델별 핵심

| 모델 | 입력/중간 shape | 핵심 객체 |
|---|---|---|
| LSTM | `[B,50,1]→[B,50,64]→[B,1]` | `nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)` |
| RNN | 동일 | `nn.RNN`, 마지막 time step을 `Linear(64,1)`에 전달 |
| Conv1D | `[B,50,1]→transpose [B,1,50]` | `Conv1d(in_channels=1, out_channels=64, kernel_size=2)` 후 flatten/FC |
| Encoder-Decoder | encoder `[B,50,1]`, decoder horizon `10` | hidden state를 decoder에 전달, 출력 `[B,10,1]` |

RNN/LSTM은 `out, hidden = layer(x)`를 반환한다. one-step 예측은 `out[:, -1, :]`을 써서 `[B,64]`로 만든 후 FC에 넣는다. Conv1D는 `Conv1d`가 channel-first를 요구하므로 `x.transpose(1,2)`가 필요하다.

## 3. 학습·평가

- loss: `nn.MSELoss(reduction="mean")`, optimizer: `Adam(lr=1e-3)`, 예시 epoch 10.
- 평가 전 `model.eval()`과 `torch.no_grad()`를 사용한다.
- 예측과 target을 원래 단위로 복원할 때 `scaler.inverse_transform`을 적용한다.
- `RMSE = sqrt(mean((y-yhat)^2))`; `MAPE = mean(abs((y-yhat)/y))`. 실제 값이 0이면 MAPE가 불안정하다.
- scaled 값의 RMSE와 원 단위 RMSE를 혼동하지 말고 출력 단위를 명시한다.

## 4. 빈칸 디버깅 순서

1. `X_train.shape[-1]`로 `input_size`를 결정한다.
2. LSTM/RNN `batch_first=True`를 확인한다.
3. Conv1D 입력 transpose와 FC의 `in_features`를 계산한다.
4. `y_hat`과 `y` shape를 맞춘다. 마지막 step이면 `y_hat[:, -1, 0]`과 `y[:, 0]`이 비교 대상이다.
5. recursive forecast에서는 매 step 예측을 window에 append하고 가장 오래된 값을 제거한다.

시간순 split 전에 shuffle하지 않는 것이 원칙이다. 미래 관측치를 사용해 window/scaler를 만들면 높은 점수라도 실제 예측 성능을 보장하지 않는다.
