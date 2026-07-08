# On-Device AI Practice 01 — Pruning for CNN 코드 기준 학습 가이드


## 그림으로 먼저 잡기

```mermaid
flowchart LR
  A["Baseline CNN"] --> B["Select pruning target"] --> C["Apply mask"] --> D["Fine-tune"] --> E["Compare accuracy / sparsity"]
```

| 코드 블록 | 눈으로 확인할 것 | 의미 |
|---|---|---|
| model 정의 | Conv/Linear layer 이름 | 어떤 weight가 pruning되는지 |
| pruning 함수 | mask buffer, zero weight | 실제 parameter가 0이 되는 위치 |
| 평가 루프 | accuracy 변화 | pruning 후 성능 손실 |

> 같이 볼 원본 노트북: `On-Device AI 강의자료/실습/1. Pruning for CNN.ipynb`  
> 핵심 목표: **VGG-style CNN에서 weight를 0으로 만들고, mask를 유지하고, fine-tuning으로 정확도를 회복하는 전체 흐름**을 직접 구현할 수 있게 만든다.

## 0. 이 실습에서 보는 큰 그림

CNN pruning은 “작은 weight는 출력에 덜 중요할 것이다”라는 가정에서 출발한다. 모델 파라미터 $W$ 중 일부를 제거하면 실제 구현에서는 보통 다음처럼 표현된다.

$$
W_{pruned} = W \odot M, \quad M_{ij} \in \{0,1\}
$$

- $W$: 원래 weight tensor
- $M$: pruning mask
- $\odot$: element-wise multiplication
- $M_{ij}=0$: 해당 연결을 끊은 것으로 취급
- $M_{ij}=1$: 해당 연결 유지

중요한 점은 **0으로 만든 것 자체가 곧 하드웨어 속도 향상을 보장하지는 않는다**는 것이다. 이 실습은 먼저 “정확도 관점에서 어떤 weight를 지워도 되는가”를 익히고, 이후 구조적 pruning과 sparse representation이 왜 필요한지 연결한다.

```mermaid
flowchart LR
  A["Dense CNN"] --> B["중요도 계산: abs(w)"]
  B --> C["Mask 생성"]
  C --> D["W = W * M"]
  D --> E["정확도/크기/희소도 평가"]
  E --> F[Fine-tuning]
  F --> E
```

## 1. Setup / CIFAR-10 / VGG 모델 로딩

노트북 초반부는 실험 환경을 고정한다.

### 코드가 하는 일

```python
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
```

이 코드는 pruning 결과를 비교 가능하게 만든다. pruning threshold가 weight 분포의 분위수에 의해 결정되기 때문에, 초기 상태와 데이터 순서가 바뀌면 결과가 흔들릴 수 있다.

CIFAR-10 입력 shape는 일반적으로 다음과 같다.

| 객체 | Shape | 의미 |
|---|---:|---|
| 이미지 1장 | `[3, 32, 32]` | RGB 3채널, 32x32 |
| mini-batch | `[B, 3, 32, 32]` | `B`개 이미지 묶음 |
| Conv weight | `[C_out, C_in, K_h, K_w]` | 출력 채널, 입력 채널, 커널 높이/너비 |
| Class logits | `[B, 10]` | CIFAR-10 class score |

`torch.hub.load(..., cifar10_vgg9_bn, pretrained=True)`는 이미 학습된 VGG9-BN 모델을 가져온다. pruning 전 기준 정확도와 모델 크기를 먼저 측정해야 pruning 후 손실을 해석할 수 있다.

## 2. 모델 크기와 정확도 평가

### `get_num_parameters(model, count_nonzero_only=False)`

이 함수는 parameter tensor의 element 수를 센다.

- `count_nonzero_only=False`: 전체 parameter 수
- `count_nonzero_only=True`: 0이 아닌 parameter 수만 계산

수학적으로는 다음과 같다.

$$
N_{dense}=\sum_l |W_l|, \quad N_{nonzero}=\sum_l \|W_l\|_0
$$

여기서 $\|W\|_0$는 엄밀한 norm은 아니지만 “0이 아닌 원소 개수”를 의미한다.

### 왜 nonzero count가 중요한가?

pruning을 해도 tensor shape 자체는 그대로다. 예를 들어 `[64,3,3,3]` Conv weight에서 절반을 0으로 만들어도 저장 tensor는 여전히 `[64,3,3,3]`이다. 그래서 실제 메모리/속도 이득을 얻으려면 둘 중 하나가 필요하다.

1. sparse format으로 저장해서 0의 위치를 압축한다.
2. structured pruning으로 channel/filter 자체를 줄여 dense 연산 shape를 작게 만든다.

이 실습의 `count_nonzero_only`는 “논리적으로 얼마나 제거했는가”를 확인하는 도구다.

## 3. 학습/평가 루프

`train()`과 `evaluate()`는 pruning 전후 정확도를 비교한다.

```python
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()
```

- forward: $\hat{y}=f(x;W)$
- loss: $\mathcal{L}(\hat{y},y)$
- backward: $\nabla_W \mathcal{L}$ 계산
- optimizer: $W \leftarrow W - \eta \nabla_W \mathcal{L}$

pruning 후 fine-tuning을 할 때 주의할 점은 **pruned weight가 다시 살아나면 안 된다**는 것이다. 그래서 mask를 계속 곱하거나 gradient를 막는 방식이 필요하다.

## 4. Weight distribution과 magnitude pruning

히스토그램은 weight 값이 0 근처에 얼마나 몰려 있는지 보여준다.

Magnitude-based pruning의 기본 가정은 다음이다.

$$
|w_i| \text{가 작으면 } y = Wx \text{에 주는 영향도 작다.}
$$

예를 들어 선형층에서

$$
y_j = \sum_i w_{ji}x_i
$$

이고 $w_{ji}\approx 0$라면, 해당 항은 출력에 미치는 기여가 작을 가능성이 높다. 물론 $x_i$가 매우 크면 예외가 생기므로 LLM pruning의 Wanda에서는 activation까지 같이 본다.

## 5. Reconstruction error

노트북의 `get_reconstuction_error()`는 두 tensor 출력의 차이를 제곱합으로 잰다.

$$
E = \sum_i (a_i-b_i)^2
$$

여기서 $a$는 원래 모델의 activation 또는 output, $b$는 pruning 후 activation 또는 output으로 보면 된다.

- error가 작다: pruning 후 표현이 원래와 비슷하다.
- error가 크다: pruning이 layer 출력 구조를 크게 흔들었다.

정확도는 최종 classification 기준이고, reconstruction error는 중간 표현이 얼마나 보존되는지 보는 기준이다.

## 6. Pruning granularity / pattern

이 실습의 핵심 비교는 fine-grained, vector-level, kernel-level, channel-level pruning이다.

| 방식 | 제거 단위 | 예시 weight shape 기준 | 장점 | 단점 |
|---|---|---|---|---|
| Fine-grained | 원소 하나 | `W[o,i,h,w]` | 정확도 손실이 작을 수 있음 | 실제 가속 어려움 |
| Vector-level | 벡터 | 특정 row/column slice | 0-vector skip 가능 | 제약이 강함 |
| Kernel-level | 커널 | `W[o,i,:,:]` | Conv 구조와 어느 정도 맞음 | fine보다 자유도 낮음 |
| Channel-level | 채널/filter | `W[o,:,:,:]` 또는 입력 채널 | dense shape 자체 축소 가능 | 정확도 손실 위험 큼 |

### Fine-grained가 hardware acceleration이 어려운 이유

fine-grained는 0의 위치가 불규칙하다. GPU/CPU는 보통 연속된 dense block을 빠르게 곱하도록 설계되어 있다.

```text
Dense GEMM:      규칙적 연속 메모리 접근 + SIMD/GPU tile 최적화
Fine sparse:     값 + index 접근 + 불규칙 분기 + load imbalance
Structured:      통째 channel/filter 제거 → 작은 dense GEMM/Conv로 변환 가능
```

따라서 fine-grained pruning은 “parameter 수”는 줄일 수 있지만, 전용 sparse kernel 또는 2:4 같은 hardware-supported pattern이 없으면 속도 향상이 제한적이다.

## 7. FineGrainedPruner / mask 유지

노트북의 pruner class는 대략 다음 책임을 가진다.

1. 각 parameter tensor에서 threshold 계산
2. `mask = abs(weight) > threshold` 생성
3. `weight *= mask` 적용
4. fine-tuning 중에도 같은 mask를 재적용

중요한 구현 감각:

```python
mask = torch.abs(weight) > threshold
weight.mul_(mask)
```

여기서 `mask`는 boolean tensor지만 곱셈 시 0/1처럼 동작한다. pruning된 weight가 optimizer update로 다시 nonzero가 될 수 있으므로, 학습 step 이후 mask를 다시 곱해야 한다.

## 8. One-shot vs Iterative pruning

### One-shot pruning

한 번에 목표 sparsity까지 제거한다.

$$
M = \mathbf{1}(|W| > \tau_s)
$$

- 빠르다.
- 정확도 손실이 클 수 있다.

### Iterative pruning

여러 epoch에 걸쳐 sparsity를 조금씩 올린다.

$$
s_t = s_f + (s_i-s_f)\left(1-\frac{t-t_0}{n\Delta t}\right)^3
$$

직관적으로는 “갑자기 많이 자르지 말고, 모델이 적응할 시간을 준다”는 뜻이다. pruning 후 fine-tuning이 중요한 이유도 여기서 나온다.

## 9. 직접 구현할 때 체크리스트

1. 기준 dense accuracy를 먼저 저장한다.
2. 각 layer weight shape를 출력해서 pruning 단위가 무엇인지 확인한다.
3. threshold는 전체 tensor 기준인지 layer별 기준인지 명확히 정한다.
4. mask shape가 weight shape와 같은지 확인한다.
5. pruning 후 nonzero count와 sparsity를 출력한다.
6. fine-tuning 중 pruned weight가 다시 살아나지 않는지 확인한다.
7. accuracy, reconstruction error, model size를 같이 본다.

## 10. 시험 대비 핵심 문장

- Pruning은 weight를 0으로 만들어 연결을 논리적으로 제거하는 과정이다.
- Fine-grained pruning은 정확도 보존에는 유리할 수 있지만 불규칙 sparse pattern 때문에 일반 hardware acceleration이 어렵다.
- Structured pruning은 channel/filter 단위로 제거하므로 실제 dense tensor shape를 줄이기 쉽다.
- Mask는 pruning된 weight를 계속 0으로 유지하기 위한 핵심 구현 장치다.
- One-shot은 빠르지만 손실이 크고, iterative pruning은 fine-tuning과 결합해 손실을 줄인다.
