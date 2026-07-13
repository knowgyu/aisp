# 시험 직전: Jupyter Skeleton Code 생존 체크리스트

시험은 객관식이 아니라 빈칸이 있는 notebook을 완성하는 형식이다. 목표는 코드를 통째로 암기하는 것이 아니라 **각 셀의 입력·출력·shape·다음 셀과의 연결**을 빠르게 복원하는 것이다.

## 1. 시작 5분: 실행 순서부터 고정

```text
imports → seed/device → data load → split/transform → Dataset/DataLoader
→ model class → train loop → eval/inference → metric/plot
```

- Kernel을 처음부터 실행할 수 있는지 확인한다.
- 앞 셀의 변수 이름과 type/shape를 출력한다. (`type(x)`, `x.shape`, `x.dtype`, `x.device`)
- `TODO`를 보기 전에 바로 위 셀과 바로 아래 셀이 요구하는 입출력을 읽는다.
- 하나의 TODO를 채운 뒤 전체 실행보다 그 셀 직후의 shape/assert부터 확인한다.

## 2. PyTorch 최소 골격

```python
model.train()
for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    pred = model(x)
```

자주 빠뜨리는 것:

- `optimizer.zero_grad()` 없으면 gradient가 누적된다.
- 평가에는 `model.eval()`과 `torch.no_grad()`를 같이 둔다.
- `pred.shape`와 `y.shape`를 맞춘 뒤 loss를 호출한다. 무심코 broadcast되는 오류를 경계한다.
- GPU tensor는 `tensor.detach().cpu().numpy()` 후 NumPy/plot에 넘긴다.

## 3. Shape를 적는 습관

| 상황 | 먼저 확인할 shape |
|---|---|
| 이미지 CNN | `[B, C, H, W]` |
| 시계열 RNN/LSTM | `[B, T, F]` (`batch_first=True`) |
| Conv1d | `[B, C, T]` — 필요하면 `transpose(1, 2)` |
| user/item embedding | id `[B]` → embedding `[B, D]` |
| graph edge | `edge_index [2, E]` |
| RAG embedding | 문서/질문 각각 고정 길이 `[D]` |

`view`, `reshape`, `squeeze`, `unsqueeze`, `transpose`를 쓸 때는 **어느 축을 없애거나 추가하는지** 주석으로 남긴다. 특히 `squeeze()`는 batch=1일 때 batch 축도 지울 수 있으므로 `squeeze(-1)`처럼 축을 지정하는 편이 안전하다.

## 4. 데이터 누수와 split

- 시계열: 시간순 split 후 train에만 scaler `fit`; test는 `transform`만.
- 추천: test edge를 graph message passing input에 넣지 않는다.
- 일반 분류: label encoder/tokenizer를 train 기준으로 맞춘다.
- test는 모델 선택·hyperparameter 조정에 쓰지 않는다.

## 5. 라이브러리별 빠른 단서

| 보이는 import/API | 바로 떠올릴 역할 |
|---|---|
| `pandas`, `numpy` | DataFrame/array 전처리 |
| `sklearn.preprocessing.MinMaxScaler` | train 통계 fit, transform |
| `torch.utils.data.Dataset`, `DataLoader` | sample → batch 공급 |
| `nn.Embedding` | 정수 ID → dense vector |
| `nn.LSTM(..., batch_first=True)` | `[B,T,F]` 입력 |
| `torch_geometric` | graph / `edge_index [2,E]` |
| `VectorStoreIndex` | Document→Node→embedding index |
| `BasicMCPClient`, `FunctionAgent` | MCP tool 목록·Agent workflow |

## 6. 제출 전 60초 체크

1. 셀을 위에서 아래로 재실행해 숨은 state 의존성이 없는지 확인한다.
2. `NameError`면 변수의 생성 셀/철자를, `AttributeError`면 객체 type을 본다.
3. shape error면 `print(x.shape, y.shape, pred.shape)`를 loss 직전에 넣는다.
4. metric 방향을 확인한다: RMSE/MAPE는 낮을수록, Precision/Recall@K는 높을수록 좋다.
5. 빈 `TODO`, 하드코딩된 경로, API key placeholder가 남지 않았는지 확인한다.
6. 결과가 이상하면 먼저 data split·target alignment·`model.eval()`을 의심한다.

## 7. 시험장에서 외울 한 줄

> **입력은 어디서 왔나 → shape는 무엇인가 → model이 무엇을 반환하나 → target과 loss가 맞나 → 평가는 누수 없이 했나.**
