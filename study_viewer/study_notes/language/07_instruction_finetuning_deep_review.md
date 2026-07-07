# LLM 07. Instruction Fine-tuning 깊은 복습

> 대상 원본: `llm_hands_on/Chapter_7_Exercise_Follow_Instructions.ipynb`  
> 목표: instruction dataset을 prompt/response 형식으로 만들고, response 영역만 학습하는 이유를 이해한다.

---

## 0. 한 장 요약

```text
instruction + input + output
 -> formatted prompt text
 -> token ids
 -> input_ids:  [B,L]
 -> target_ids: [B,L]
 -> prompt 부분 target은 -100으로 masking
 -> response token만 loss 계산
```

---

## 1. Instruction data 형식

보통 한 sample은 다음 구조다.

```json
{
  "instruction": "Translate the sentence to Korean.",
  "input": "I like apples.",
  "output": "나는 사과를 좋아합니다."
}
```

format 함수는 이를 모델 입력 문자열로 만든다.

```text
Below is an instruction...
### Instruction:
...
### Input:
...
### Response:
...
```

---

## 2. 왜 formatting이 중요한가?

모델은 단순히 token sequence를 본다. “여기부터 지시문, 여기부터 답변”이라는 구조를 문자열 패턴으로 배운다.

```text
좋은 format = 학습/추론 때 일관된 구분자
나쁜 format = 모델이 어디에 답해야 하는지 혼란
```

---

## 3. target masking `-100`

PyTorch `CrossEntropyLoss(ignore_index=-100)`는 target이 -100인 위치의 loss를 무시한다.

```text
input tokens:  [prompt tokens ... response tokens]
target tokens: [-100, -100, ... response target ids]
```

왜 prompt를 학습하지 않나?

- prompt는 사용자가 제공하는 조건이다.
- 우리가 원하는 것은 response를 잘 생성하는 것이다.
- prompt token까지 예측하도록 loss를 주면 “질문을 외우는” 비중이 생긴다.

---

## 4. Collate function

서로 길이가 다른 sample을 batch로 묶기 위해 padding한다.

```text
sample1 length 120
sample2 length 80
-> pad sample2 to 120
```

shape:

```text
input_ids:  [B,Lmax]
target_ids: [B,Lmax]
```

padding target도 보통 `-100`으로 둬 loss에서 제외한다.

---

## 5. 학습 loss

GPT pretraining과 같은 next-token loss지만, target mask가 다르다.

```text
logits: [B,L,V]
target: [B,L]
flatten -> [B*L,V], [B*L]
ignore_index=-100
```

---

## 6. 추론

추론 때는 instruction+input까지만 넣고 response를 생성한다.

```text
prompt = format_input(entry) + "\n### Response:\n"
model.generate(prompt)
```

학습 때 본 format과 추론 format이 같아야 한다.

---

## 7. 복습 질문

1. instruction fine-tuning도 본질적으로 next-token prediction인가?
2. target의 `-100`은 어떤 역할인가?
3. prompt 영역 loss를 무시하는 이유는?
4. 학습 format과 추론 format이 다르면 어떤 문제가 생기는가?
