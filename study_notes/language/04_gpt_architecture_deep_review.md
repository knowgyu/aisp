# LLM 04. GPT Architecture 깊은 복습

> 대상 원본: `llm_hands_on/Chapter_4_Excercise_GPT.ipynb`  
> 목표: GPT 블록을 Dataset/Attention/LayerNorm/FFN/Generation까지 shape로 연결한다.

---

## 0. GPT 전체 흐름

```text
idx [B,T]
 -> token embedding [B,T,D]
 -> positional embedding 더하기 [B,T,D]
 -> TransformerBlock x L [B,T,D]
 -> final LayerNorm [B,T,D]
 -> output head Linear(D -> vocab) [B,T,V]
```

다음 token 예측은 각 위치마다 vocab 전체 logit을 낸다.

---

## 1. 설정값 의미

예: GPT-2 Small 스타일

| key | 예 | 의미 |
|---|---:|---|
| `vocab_size` | 50257 | token 종류 수 |
| `context_length` | 1024 | 최대 sequence 길이 |
| `emb_dim` | 768 | token vector 차원 |
| `n_heads` | 12 | attention head 수 |
| `n_layers` | 12 | Transformer block 수 |
| `drop_rate` | 0.1 | dropout 비율 |

---

## 2. LayerNorm

LayerNorm은 마지막 차원 `D`를 정규화한다.

```text
x: [B,T,D]
mean/var over D
output: [B,T,D]
```

BatchNorm과 달리 batch 통계가 아니라 token별 feature 차원 통계다. Transformer에 적합하다.

---

## 3. GELU

GPT FFN은 ReLU 대신 GELU를 많이 쓴다.

```text
GELU(x) ≈ x * Φ(x)
```

작은 음수도 완전히 죽이지 않아 smoother한 activation이다.

---

## 4. FeedForward

```python
nn.Sequential(
    nn.Linear(D, 4*D),
    GELU(),
    nn.Linear(4*D, D),
)
```

shape:

```text
[B,T,D] -> [B,T,4D] -> [B,T,D]
```

Attention이 token 간 정보 혼합, FFN은 token별 feature 변환이다.

---

## 5. TransformerBlock

GPT block:

```text
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

Residual connection이 있어 각 block은 “기존 표현 + 수정량”을 학습한다.

---

## 6. GPTModel forward

```python
tok_embeds = self.tok_emb(idx)      # [B,T,D]
pos_embeds = self.pos_emb(arange(T))# [T,D]
x = tok_embeds + pos_embeds        # [B,T,D]
x = self.drop_emb(x)
x = self.trf_blocks(x)             # [B,T,D]
x = self.final_norm(x)             # [B,T,D]
logits = self.out_head(x)          # [B,T,V]
```

중요: `logits[:, -1, :]`는 마지막 token 위치의 다음 token 분포다.

---

## 7. generate_text_simple

```python
for _ in range(max_new_tokens):
    idx_cond = idx[:, -context_size:]
    logits = model(idx_cond)
    logits = logits[:, -1, :]
    probas = torch.softmax(logits, dim=-1)
    idx_next = torch.argmax(probas, dim=-1, keepdim=True)
    idx = torch.cat((idx, idx_next), dim=1)
```

shape:

```text
idx:       [B,T]
idx_cond:  [B,min(T,context)]
logits:    [B,T_cond,V]
last:      [B,V]
idx_next:  [B,1]
new idx:   [B,T+1]
```

argmax는 deterministic greedy decoding이다.

---

## 8. 복습 질문

1. GPT 출력 `[B,T,V]`에서 `V`는 무엇인가?
2. 다음 token 생성 때 왜 `logits[:, -1, :]`만 쓰는가?
3. FFN의 중간 차원이 보통 `4D`인 이유는 무엇인가?
4. LayerNorm은 `[B,T,D]` 중 어느 축을 정규화하는가?
