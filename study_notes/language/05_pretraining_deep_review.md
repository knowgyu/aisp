# LLM 05. Pretraining 깊은 복습

> 대상 원본: `llm_hands_on/Chapter_5_Excercise_Pretraining.ipynb`  
> 목표: next-token pretraining loop, loss 계산, sampling을 코드 shape로 이해한다.

---

## 0. Pretraining 목표

언어모델 pretraining은 다음 token 예측이다.

```text
input:  [t0, t1, t2, ..., tL-1]
target: [t1, t2, t3, ..., tL]
```

모델은 각 위치에서 다음 token의 vocab 분포를 낸다.

```text
logits: [B,L,V]
target: [B,L]
```

---

## 1. loss 계산

노트북 핵심:

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )
    return loss
```

shape:

```text
logits: [B,L,V]
logits.flatten(0,1): [B*L,V]
target: [B,L]
target.flatten(): [B*L]
```

CrossEntropyLoss는 각 token position을 하나의 classification 문제로 본다.

---

## 2. train_model_simple

```text
for epoch:
  for input_batch, target_batch:
    optimizer.zero_grad()
    loss = calc_loss_batch(...)
    loss.backward()
    optimizer.step()
```

`examples_seen`은 처리한 sample 개수, `global_step`은 optimizer update 횟수다.

---

## 3. 평가

학습 중 일정 step마다 train/val loss를 계산한다.

```text
train loss 감소: 모델이 학습 데이터의 다음 token을 잘 맞춤
val loss 감소: 일반화도 개선
train만 감소하고 val 증가: overfitting
```

---

## 4. 텍스트 생성: temperature와 top-k

기본 greedy는 항상 가장 큰 logit을 고른다. 다양성이 낮다.

Temperature:

```text
logits = logits / temperature
```

| temperature | 효과 |
|---:|---|
| `< 1` | 분포가 날카로워짐, 보수적 |
| `= 1` | 원래 분포 |
| `> 1` | 분포가 평평해짐, 다양 |

Top-k:

```text
가장 높은 k개 token만 남기고 나머지는 -inf
```

둘을 같이 쓰면 말이 안 되는 낮은 확률 token을 막으면서도 다양성을 준다.

---

## 5. text_to_token_ids / token_ids_to_text

```python
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded).unsqueeze(0)
```

shape:

```text
encoded list length T
Tensor: [T]
unsqueeze(0): [1,T]
```

batch dimension을 추가해야 모델 입력 `[B,T]`와 맞는다.

---

## 6. 복습 질문

1. `logits.flatten(0,1)`는 왜 필요한가?
2. pretraining target은 원문 token과 몇 칸 shift되는가?
3. temperature가 낮으면 생성은 더 무작위가 되는가?
4. top-k는 어떤 token들을 제거하는가?
