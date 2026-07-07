# LLM 06+. LoRA Fine-tuning 깊은 복습

> 대상 원본: `llm_hands_on/Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb`  
> 목표: LoRA가 왜 parameter-efficient fine-tuning인지, 코드에서 어떤 weight가 학습되는지 이해한다.

---

## 0. 한 장 요약

LoRA는 기존 weight `W`를 직접 수정하지 않고 low-rank update `BA`만 학습한다.

```text
original: y = x W
LoRA:     y = x W + alpha/r * x A B
```

| 항목 | shape 예 |
|---|---:|
| 원 weight `W` | `[in_dim, out_dim]` |
| LoRA A | `[in_dim, r]` |
| LoRA B | `[r, out_dim]` |
| rank `r` | 작은 값, 예: 8 |

---

## 1. 왜 LoRA인가?

full fine-tuning은 모든 파라미터를 업데이트한다. 큰 LLM에서는 비용이 크다.

LoRA는 대부분 weight를 freeze하고 작은 adapter만 학습한다.

```text
학습되는 파라미터 수 ≈ r*(in_dim + out_dim)
원 Linear 파라미터 수 = in_dim*out_dim
```

예: `in=768, out=768, r=8`

```text
원래: 768*768 = 589,824
LoRA: 8*(768+768) = 12,288
약 2.1%
```

---

## 2. LoRALayer 구조

일반 구현:

```python
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)
```

초기 `B=0`이면 시작 시 LoRA branch 출력이 0이라 원 모델 출력을 보존한다.

---

## 3. LinearWithLoRA

```python
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
    def forward(self, x):
        return self.linear(x) + self.lora(x)
```

shape:

```text
x:           [..., in_dim]
linear(x):   [..., out_dim]
lora(x):     [..., out_dim]
sum:         [..., out_dim]
```

---

## 4. 어디에 LoRA를 붙이나?

Transformer에서 보통 attention projection에 붙인다.

| 위치 | 의미 |
|---|---|
| `W_query` | 어떤 정보를 찾을지 |
| `W_key` | 어떤 정보로 매칭될지 |
| `W_value` | 어떤 내용을 가져올지 |
| `out_proj` | head concat 후 출력 projection |
| FFN linear | token feature 변환 |

실습에서는 선택한 `nn.Linear`를 `LinearWithLoRA`로 교체한다.

---

## 5. freeze 확인

```python
for param in model.parameters():
    param.requires_grad = False
```

그 뒤 LoRA parameter만 `requires_grad=True`.

체크:

```text
trainable params / total params
```

LoRA의 핵심 검증은 “학습 가능한 parameter가 정말 줄었는지”다.

---

## 6. 복습 질문

1. LoRA에서 원래 weight `W`는 학습되는가?
2. `B`를 0으로 초기화하면 시작 출력이 왜 원 모델과 같은가?
3. rank `r`이 커지면 표현력과 비용은 어떻게 변하는가?
4. `x @ A @ B`의 최종 shape은 왜 원 Linear 출력과 같아야 하는가?
