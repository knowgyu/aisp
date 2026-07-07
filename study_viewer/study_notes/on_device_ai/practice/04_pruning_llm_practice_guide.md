# On-Device AI Practice 04 — Pruning for LLM 코드 기준 학습 가이드

> 같이 볼 원본 노트북: `On-Device AI 강의자료/실습/4. Pruning for LLM.ipynb`  
> 핵심 목표: **LLM Linear layer의 weight를 magnitude/Wanda 기준으로 제거하고 perplexity 변화로 평가하는 흐름**을 이해한다.

## 0. CNN pruning과 LLM pruning의 차이

LLM에서 대부분의 계산은 Linear layer의 matrix multiplication이다.

$$
y = xW^T
$$

Transformer block 내부의 attention projection, MLP up/down/gate projection이 모두 큰 matrix다. 그래서 LLM pruning은 보통 `nn.Linear` weight를 대상으로 한다.

| 항목 | CNN pruning | LLM pruning |
|---|---|---|
| 대표 layer | Conv2d | Linear |
| 평가 지표 | accuracy | perplexity |
| 입력 구조 | image `[B,C,H,W]` | token ids `[B,T]` |
| 주요 이슈 | channel/filter 구조 | activation-aware importance |

## 1. Perplexity 평가

노트북의 `evaluate(model, tokenizer)`는 Wikitext-2로 perplexity를 계산한다.

언어모델 loss는 next-token negative log likelihood다.

$$
\mathcal{L}=-\frac{1}{N}\sum_{t}\log p(x_t|x_{<t})
$$

Perplexity는 다음이다.

$$
PPL=e^{\mathcal{L}}
$$

- 낮을수록 좋다.
- pruning으로 모델 예측이 나빠지면 loss가 증가하고 PPL이 오른다.

## 2. SmolLM-135M 로딩

`AutoModelForCausalLM`은 causal language model을 로드한다. tokenizer는 문자열을 token id로 바꾼다.

| 객체 | Shape | 의미 |
|---|---:|---|
| `input_ids` | `[B,T]` | token id sequence |
| embedding output | `[B,T,d_model]` | token vector |
| Linear weight | `[d_out,d_in]` | projection matrix |
| logits | `[B,T,V]` | vocabulary score |

`lm_head`는 마지막 vocabulary projection이다. pruning 코드에서 `lm_head`를 제외하는 이유는 출력 vocabulary 분포가 직접 손상될 수 있고, tied embedding 구조와도 관련될 수 있기 때문이다.

## 3. Magnitude-based pruning

노트북의 `prune_magnitude(model, sparsity)`는 `nn.Linear` weight에 대해 작은 값을 제거한다.

기본 아이디어:

$$
M_{ij}=\mathbf{1}(|W_{ij}|>\tau)
$$

$$
W' = W\odot M
$$

구현 감각:

```python
threshold = torch.quantile(torch.abs(weight), sparsity)
mask = torch.abs(weight) > threshold
weight.mul_(mask)
```

`sparsity=0.5`라면 절반 정도의 weight가 0이 된다.

### LLM에서 magnitude만 보면 부족한 이유

출력 변화는 weight만으로 정해지지 않는다.

$$
y_j=\sum_i W_{ji}x_i
$$

$W_{ji}$가 작아도 $x_i$가 매우 자주 크면 영향이 커질 수 있다. 그래서 Wanda는 activation 크기를 같이 본다.

## 4. Calibration dataset

Wanda는 calibration 데이터로 activation 통계를 모은다.

`get_calib_dataset()`은 여러 text sample을 가져와 token block을 만든다. 목적은 실제 inference에서 들어올 법한 $x$ 분포를 보는 것이다.

| 변수 | 의미 |
|---|---|
| `n_samples` | calibration sample 수 |
| `block_size` | token sequence length |
| `input_ids` | 모델에 넣는 token ids |
| activation | 특정 Linear layer 입력 $x$ |

Calibration은 학습이 아니다. weight를 업데이트하지 않고 activation 통계만 수집한다.

## 5. Hook으로 activation 수집

`get_calib_feat()`는 forward hook을 이용해 각 Linear module의 입력을 저장한다.

개념적으로는:

```python
def hook(module, x, y):
    input_feat[name] += x[0]
```

Linear layer 입력 shape는 보통 `[B,T,d_in]`이다. 여러 sample/token에 대해 feature norm을 누적한다.

## 6. Wanda pruning

Wanda의 중요도는 weight magnitude와 activation norm을 곱한다.

$$
S_{ij}=|W_{ij}|\cdot \|X_j\|_2
$$

- $|W_{ij}|$: connection 자체의 크기
- $\|X_j\|_2$: 해당 input channel이 calibration에서 얼마나 활성화되는지

즉 “작은 weight”뿐 아니라 “잘 쓰이지 않는 input dimension에 연결된 weight”를 우선 제거한다.

### Row-wise pruning

Wanda는 일반적으로 row별로 sparsity를 맞춘다.

```text
W shape [d_out, d_in]
각 output row마다 d_in 중 일부 제거
```

row-wise로 하면 output neuron마다 일정 비율의 입력 연결이 유지된다. 특정 row가 과도하게 망가지는 것을 줄인다.

## 7. 직접 구현 체크리스트

1. `nn.Linear`만 대상으로 순회한다.
2. `lm_head` 제외 여부를 명시한다.
3. magnitude pruning은 threshold 기준을 global/layer/row 중 무엇으로 할지 정한다.
4. Wanda는 calibration forward에서 gradient를 끈다.
5. hook을 등록한 뒤 반드시 제거한다.
6. activation norm shape가 weight의 input dimension과 broadcast되는지 확인한다.
7. pruning 후 PPL을 dense baseline과 비교한다.
8. GPU memory를 비우는 `gc.collect()`, `torch.cuda.empty_cache()` 위치를 확인한다.

## 8. 시험 대비 핵심 문장

- LLM pruning은 Transformer의 큰 Linear matrix를 주로 대상으로 한다.
- Perplexity는 next-token prediction loss의 exponential이며 낮을수록 좋다.
- Magnitude pruning은 $|W|$만 보지만, Wanda는 $|W|$와 activation norm을 함께 본다.
- Calibration dataset은 학습용이 아니라 activation statistics 추정용이다.
- Wanda의 row-wise pruning은 output dimension별 손상을 균형 있게 만드는 효과가 있다.
