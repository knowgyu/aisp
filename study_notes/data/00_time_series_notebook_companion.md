# Time Series Notebook Companion: window와 순서 보존

원본: `data/3_4-ts-practice/ts_practice.ipynb`, 해답: `ts_solution.ipynb`. 이 문서는 노트북의 셀을 shape와 객체 책임 중심으로 복습한다.

## 1. 전체 파이프라인

```text
raw series [N,F]
 → 시간 정렬·결측/스케일 처리
 → sliding window: X [samples,T,F], y [samples,H]
 → Dataset/DataLoader
 → RNN/LSTM/GRU output [B,T,Hid]
 → 마지막 hidden 또는 sequence head
 → Linear [Hid,H] → prediction [B,H]
```

`T`는 lookback 길이, `F`는 feature 수, `H`는 예측 horizon이다. 단변량 one-step이면 `F=1,H=1`; 다변량·multi-step이면 target shape를 반드시 명시한다.

## 2. 코드 객체 역할

| 객체 | 역할 |
|---|---|
| window maker | `series[i:i+T]`와 미래 target을 생성; 미래 구간이 입력에 섞이지 않게 함 |
| `StandardScaler`/정규화 | train 구간으로 fit, validation/test에는 transform만 적용 |
| `TensorDataset` | `X`, `y` tensor를 같은 index로 보관 |
| `DataLoader` | batch `[B,T,F]`; train shuffle은 문제 설정에 따라 결정 |
| `nn.LSTM(batch_first=True)` | 입력 `[B,T,F]`, 출력 sequence `[B,T,Hid]`, hidden tuple |
| `nn.Linear(Hid,H)` | 마지막 hidden을 horizon 예측으로 변환 |
| `MSELoss`/`L1Loss` | 예측과 target shape를 맞춰 회귀 손실 계산 |

## 3. 평가와 누수 방지

시간 순서대로 train→validation→test를 나눈다. scaler는 train에만 fit한다. `model.eval()`과 `torch.no_grad()`로 dropout/gradient를 끄고, inverse transform 후 MAE/RMSE를 원래 단위에서 보고한다. baseline(naive: 마지막 값 유지)보다 좋아지는지도 확인한다.

- MAE: `mean(abs(y - yhat))`, 해석이 쉽고 낮을수록 좋다.
- MSE: 큰 오차를 강하게 벌점, 낮을수록 좋다.
- RMSE: MSE의 원래 단위 제곱근, 낮을수록 좋다.

## 4. 시험형 shape 질문

- `batch_first=True`가 없으면 기본 입력은 `[T,B,F]`가 될 수 있다.
- `output[:, -1, :]`는 마지막 시점 hidden `[B,Hid]`이다.
- multi-step head는 `[B,Hid] → [B,H]`; target도 `[B,H]`여야 한다.
- `squeeze()`는 batch size 1일 때 batch 축까지 제거할 수 있으므로 `squeeze(-1)`처럼 축을 지정한다.

## 5. 노트북별 구현 포인트

실습의 단변량 기본 입력은 `(N, seq_len, 1)`이다. RNN/LSTM은 sequence output을 사용하고, 1D CNN은 시간축 convolution을 적용한다. encoder-decoder는 입력 sequence를 context로 압축한 뒤 horizon 길이의 sequence를 복원하므로 `[B, seq_len, 1] → [B, H, 1]` shape를 확인한다. TODO 빈칸에서는 window index, `forward`의 마지막 시점 선택, loss와 inverse transform 순서를 우선 점검한다.
