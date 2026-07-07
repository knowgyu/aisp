# LLM 06. Classification Fine-tuning 깊은 복습

> 대상 원본: `llm_hands_on/Chapter_6_Excercise_Finetuning_Classification.ipynb`  
> 목표: GPT를 문장 분류기로 바꾸는 방식과 마지막 token logit 사용 이유를 이해한다.

---

## 0. 한 장 요약

```text
text -> tokenizer -> padded token ids [B,L]
GPT -> hidden/logits per position [B,L,num_classes]
마지막 token 위치만 사용 -> [B,num_classes]
CrossEntropyLoss with labels [B]
```

---

## 1. 데이터: SMS spam classification

원본 데이터는 `Label`, `Text` 컬럼을 가진다.

| Label | Text |
|---|---|
| ham | normal message |
| spam | spam message |

binary classification이므로 label은 보통:

```text
ham -> 0
spam -> 1
```

---

## 2. balanced dataset

스팸 데이터는 보통 적다. `create_balanced_dataset`은 ham을 spam 개수만큼 downsample한다.

```text
before: ham 4825, spam 747
 after: ham 747, spam 747
```

목표는 accuracy가 다수 class만 찍어도 높아지는 문제를 줄이는 것이다.

---

## 3. SpamDataset shape

```python
encoded_texts = [tokenizer.encode(text) for text in self.data['Text']]
encoded_text = encoded_text[:max_length]
encoded_text = encoded_text + [pad_token_id] * (max_length - len(encoded_text))
```

`__getitem__`:

```text
input_ids: [L]
label: scalar
```

DataLoader batch:

```text
input_batch:  [B,L]
target_batch: [B]
```

---

## 4. 왜 padding token이 필요한가?

배치 tensor는 rectangular해야 한다.

```text
문장 A 길이 12
문장 B 길이 48
문장 C 길이 23
```

그대로는 `[B,L]` tensor가 안 된다. 가장 긴 길이나 지정 max_length에 맞춰 padding한다.

```text
[A tokens ... PAD PAD]
[B tokens ...]
[C tokens ... PAD]
```

GPT-2에서는 `<|endoftext|>` id `50256`을 pad로 재사용하는 경우가 많다.

---

## 5. 모델 head 바꾸기

pretrained GPT는 vocab logits `[B,L,V]`를 낸다. classification에서는 class logits `[B,L,2]`가 필요하다.

```text
out_head: Linear(D -> vocab_size)
분류용으로 Linear(D -> 2)로 교체
```

---

## 6. 마지막 token만 쓰는 이유

```python
logits = model(input_batch)[:, -1, :]
```

shape:

```text
model(input_batch): [B,L,2]
[:, -1, :]:         [B,2]
```

GPT는 causal model이라 마지막 위치의 hidden state가 앞 token들을 모두 볼 수 있다.

```text
position L-1 representation = text prefix 전체를 요약한 표현
```

그래서 마지막 token 위치를 sequence representation처럼 쓴다.

주의: padding이 있으면 진짜 마지막 non-pad token이 아니라 padded 마지막 위치를 쓰는 문제가 생길 수 있다. 이 실습은 단순화를 위해 fixed padded 마지막을 사용한다. 실무에서는 attention mask와 마지막 non-pad index를 쓰는 편이 더 안전하다.

---

## 7. loss/accuracy

```text
logits: [B,2]
target: [B]
loss = cross_entropy(logits, target)
pred = argmax(logits, dim=-1)
accuracy = (pred == target).mean()
```

---

## 8. Fine-tuning 전략

| 전략 | 설명 |
|---|---|
| full fine-tuning | 모든 weight 업데이트 |
| head-only | classifier head만 학습 |
| selective unfreeze | 마지막 block 일부 + head 학습 |
| LoRA | 원 weight 고정, low-rank adapter 학습 |

이 노트북은 분류 작업에 GPT를 맞추는 기본 흐름을 보여준다.

---

## 9. 복습 질문

1. 분류 fine-tuning의 target shape은 `[B,L]`인가 `[B]`인가?
2. 왜 마지막 token 위치의 logits를 쓰는가?
3. padding이 있을 때 마지막 위치 사용은 어떤 문제가 있을 수 있는가?
4. vocab head와 classification head는 출력 차원이 어떻게 다른가?
