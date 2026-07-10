# On-Device AI Practice 01 — Pruning for CNN 코드 학습 가이드

<!-- aisp-exam-practice-notice -->
> **시험 대비 모드:** 오른쪽에는 원본 전체 코드 대신 핵심 셀을 `## 정답 입력`으로 비운 실습본이 표시됩니다.
> 원본은 `On-Device AI 강의자료/실습/1. Pruning for CNN.ipynb`, 정답과 출제 의도는 `study_notes/exam_answers/on_device_ai_code_answers.md`에서 확인합니다.


> 오른쪽 원본 노트북 `1. Pruning for CNN.ipynb`를 보면서 왼쪽에서 셀 범위를 따라가면 된다. 이 가이드는 CNN pruning 전용이며 LLM/KD/quantization 설명을 섞지 않는다.

- 기준 교안: `ODAI-1 Chapter 2. Network Pruning`
- 원본 노트북: `On-Device AI 강의자료/실습/1. Pruning for CNN.ipynb`
- 핵심 목표: VGG/CIFAR-10 모델에서 **pruning granularity → pruning ratio/sensitivity → global magnitude pruning → pruning schedule + fine-tuning**을 직접 구현 관점으로 이해한다.

## 0. 전체 흐름

```mermaid
flowchart LR
  A["Dense VGG baseline"] --> B["weight 분포와 기준 성능"]
  B --> C["fine / vector / kernel / channel pruning"]
  C --> D["sensitivity scan"]
  D --> E["layer-wise vs global pruning"]
  E --> F["one-shot vs iterative schedule"]
  F --> G["fine-tuning으로 정확도 회복"]
```

### 계속 붙잡을 수식

Pruning은 weight 자체를 학습하는 방법이 아니라, weight에 mask를 곱해 일부 연결을 제거하는 방식으로 볼 수 있다.

$$
W_{pruned}=W\odot M, \qquad M_i\in\{0,1\}
$$

- $M_i=1$: 해당 weight 유지
- $M_i=0$: 해당 weight 제거
- sparsity: 전체 원소 중 0인 비율

### 핵심 shape 표

| 대상 | Shape | 의미 |
|---|---|---|
| CIFAR-10 input | `[B, 3, 32, 32]` | RGB 이미지 batch |
| Conv weight | `[C_out, C_in, K_h, K_w]` | pruning granularity가 적용되는 기본 tensor |
| fine-grained mask | weight와 동일 | 개별 weight 단위 제거 |
| vector/kernel/channel mask | weight에 broadcast 가능 | 구조 단위 제거 |
| logits | `[B, 10]` | CIFAR-10 class score |

## 1. 노트북 전체 지도

| 구간 | 셀 범위 | 주제 | 공부 포인트 |
|---:|---:|---|---|
| 1 | 001-015 | Setup / CIFAR-10 / VGG / baseline | 기준 모델 크기와 정확도 확보 |
| 2 | 016-020 | weight distribution / reconstruction error | 작은 weight 제거 가정과 출력 변화 측정 |
| 3 | 021-039 | pruning granularity | fine, vector, kernel, channel 단위 차이 |
| 4 | 040-052 | pruning ratio / sensitivity | layer별 민감도와 non-uniform pruning |
| 5 | 053-059 | global magnitude pruning | layer-wise 기준과 global threshold 기준 비교 |
| 6 | 060-071 | pruning schedule | one-shot, cubic/linear schedule, iterative fine-tuning |

---

## 2. 구간별 Walkthrough

## Cells 001-006 — 목표, import, seed

노트북은 CNN pruning의 세 축을 선언한다.

1. 어떤 단위로 자를 것인가: granularity
2. 얼마나 자를 것인가: ratio/sparsity
3. 언제 자를 것인가: schedule

Cell 005-006은 import와 seed 고정이다. seed는 threshold나 학습 순서 재현에 필요하다.

## Cells 007-010 — CIFAR-10과 VGG baseline

CIFAR-10은 `[3,32,32]` 이미지 10개 class 분류 문제다. VGG 모델은 Conv feature extractor와 classifier로 구성된다.

오른쪽에서 확인할 것:

- DataLoader batch shape가 `[B,3,32,32]`인지.
- `model.backbone` 안의 Conv layer 이름이 무엇인지.
- `torch.hub.load`로 불러온 모델이 pretrained인지.

Pruning 전에는 반드시 dense baseline을 고정해야 한다. 이후 정확도 하락을 baseline 대비로 해석하기 때문이다.

## Cells 011-015 — 모델 크기와 정확도 평가

`get_num_parameters`는 전체 parameter 수 또는 nonzero parameter 수를 센다.

$$
N_{nonzero}=\sum_l \|W_l\|_0
$$

`get_model_size`는 parameter 수와 data width로 크기를 계산한다.

$$
\mathrm{size}=N\times \mathrm{bits\;per\;parameter}
$$

`train`과 `evaluate`는 pruning 후 fine-tuning/평가에 계속 재사용된다. pruning 실험에서 `evaluate`는 “압축이 성능을 얼마나 망가뜨렸는가”를 판단하는 기준이다.

## Cells 016-020 — weight 분포와 reconstruction error

Weight histogram은 값이 0 근처에 얼마나 몰려 있는지 보여준다. magnitude pruning의 기본 가정은 다음이다.

$$
|w_i| \text{가 작으면 출력에 주는 영향도 작을 가능성이 높다.}
$$

`get_reconstruction_error`는 원본 출력과 pruning 후 출력의 제곱 오차를 계산한다.

$$
E=\sum_i (y_i-\hat y_i)^2
$$

이 값은 정확도 평가보다 더 국소적인 비교다. 특정 layer weight만 바꿨을 때 output이 얼마나 변하는지 확인하는 용도다.

## Cells 021-027 — pruning pattern 시각화 준비

여기서는 `model.backbone.conv1.weight`를 꺼내 pruning 단위를 시각화한다. Conv weight shape는 보통 `[C_out,C_in,K_h,K_w]`이다.

예를 들어 `[128,64,3,3]`이면:

- output channel 128개
- input channel 64개
- 각 kernel은 3x3

Pruning 단위를 이해하려면 tensor 축이 무엇인지 먼저 알아야 한다.

## Cells 028-029 — Fine-grained pruning

Fine-grained pruning은 개별 weight 단위로 작은 값을 0으로 만든다.

구현 흐름:

1. `importance = abs(weight)`
2. 목표 sparsity에 해당하는 threshold 계산
3. `mask = importance > threshold`
4. `weight *= mask`

장점은 같은 sparsity에서 정확도 손실이 상대적으로 작을 수 있다는 것이다. 단점은 0 위치가 불규칙해서 일반 dense Conv kernel에서는 바로 빨라지지 않는다는 점이다.

## Cells 030-031 — Vector-level pruning

Vector-level pruning은 weight 일부 축을 벡터로 묶고 L1 norm으로 중요도를 계산한다.

$$
score(v)=\sum_i |v_i|
$$

작은 score의 vector 전체를 0으로 만든다. fine-grained보다 구조적이지만, 어떤 vector 정의를 쓰는지에 따라 hardware 효율이 달라진다.

## Cells 032-033 — Kernel-level pruning

Kernel-level은 Conv의 작은 2D kernel 단위를 제거한다. 예를 들어 특정 `(out_channel, in_channel)` 사이의 `K_h x K_w` kernel 전체를 제거한다.

- fine-grained보다 구조적이다.
- 한 번에 제거되는 정보량이 커 정확도 손실 가능성도 커진다.
- kernel 단위가 0이면 일부 연산을 더 쉽게 건너뛸 수 있다.

## Cells 034-035 — Channel-level pruning

Channel-level pruning은 입력 또는 출력 channel 전체를 제거하는 구조적 pruning이다.

실제 speedup에는 channel-level이 가장 유리한 편이다. channel을 완전히 제거하면 다음 layer의 tensor shape 자체를 줄일 수 있기 때문이다. 단, 이 노트북에서는 weight를 0으로 만드는 방식이므로 실제 shape surgery까지 하는 것은 아니다.

## Cells 036-039 — 중요도 합과 reconstruction error 비교

여기서는 네 pruning 방식이 얼마나 많은 중요도 점수를 보존하는지 비교하고, 같은 input sample에 대해 output reconstruction error를 계산한다.

해석 방법:

| 결과 | 의미 |
|---|---|
| importance sum이 높음 | 큰 weight를 많이 유지했다는 뜻 |
| reconstruction error가 낮음 | 해당 layer 출력이 원본과 비슷함 |
| accuracy가 높음 | 전체 task 성능이 유지됨 |

세 값이 항상 같은 결론을 주지는 않는다. 최종 판단은 accuracy와 실제 효율을 같이 봐야 한다.

## Cells 040-045 — sensitivity scan

Layer별로 같은 sparsity를 적용해도 정확도 하락이 다르다. sensitivity scan은 한 layer만 sparsity를 바꾸고 accuracy 변화를 측정한다.

```text
for layer in prunable_layers:
  for sparsity in [0.4, 0.5, ..., 0.9]:
    prune only that layer
    evaluate accuracy
    recover model
```

민감한 layer는 조금만 잘라도 정확도가 크게 떨어진다. 덜 민감한 layer는 더 많이 잘라도 된다.

## Cells 046-052 — layer-wise pruning과 custom sparsity

`FineGrainedPruner`는 layer별 sparsity dictionary를 받아 mask를 생성하고 적용한다.

중요한 구현 포인트:

- mask는 parameter와 같은 shape다.
- optimizer step 후 pruned weight가 다시 살아날 수 있으므로 mask를 재적용해야 한다.
- 실험마다 원본 모델로 복구해야 공정한 비교가 된다.

Uniform pruning은 모든 layer에 같은 sparsity를 적용한다. Sensitivity 기반 pruning은 layer별로 다른 sparsity를 준다.

## Cells 053-059 — global magnitude pruning

Global magnitude pruning은 layer별 threshold가 아니라 전체 weight를 한데 모아 threshold를 정한다.

$$
\tau = \mathrm{quantile}(\{|W_l|\}_{l=1}^{L}, s)
$$

장점:

- layer마다 sparsity를 사람이 정하지 않아도 된다.
- 작은 weight가 많은 layer가 더 많이 pruning된다.

주의:

- 특정 layer가 과도하게 잘리면 정확도 손실이 클 수 있다.
- sensitivity 정보를 반영하지 않는 naive global 방식은 위험할 수 있다.

## Cells 060-063 — one-shot pruning + fine-tuning

One-shot pruning은 목표 sparsity까지 한 번에 자른 뒤 fine-tuning한다. 빠르지만 정확도 충격이 크다.

Fine-tuning 루프에서 확인할 것:

1. pruning 적용
2. train epoch 수행
3. mask 재적용
4. validation accuracy 평가

## Cells 064-066 — sparsity scheduler

스케줄러는 epoch에 따라 sparsity를 점진적으로 올린다.

대표 cubic schedule:

$$
s_t=s_f+(s_i-s_f)\left(1-\frac{t-t_0}{n\Delta t}\right)^3
$$

직관은 “처음부터 많이 자르지 말고, 모델이 적응할 시간을 준다”이다.

## Cells 067-071 — iterative pruning

Iterative pruning은 pruning과 fine-tuning을 반복한다.

```text
for epoch:
  target_sparsity = scheduler(epoch)
  prune to target_sparsity
  train one epoch
  evaluate
```

일반적으로 one-shot보다 안정적으로 정확도를 회복할 수 있다. 하지만 학습 시간이 더 든다.

---

## 3. 직접 구현 체크리스트

1. baseline dense accuracy를 먼저 저장한다.
2. pruning 대상 parameter 이름과 shape를 출력한다.
3. threshold 기준이 layer-wise인지 global인지 명확히 한다.
4. mask shape가 weight shape와 같은지 확인한다.
5. sparsity는 `1 - nonzero / total`로 계산한다.
6. pruning 후 optimizer가 weight를 되살리지 않도록 mask를 재적용한다.
7. structured pruning은 “0으로 만들기”와 “실제 shape 줄이기”가 다르다는 점을 구분한다.
8. speedup을 말하려면 sparse kernel 또는 구조적 shape 감소가 필요하다.

## 4. 시험 대비 핵심 문장

- Fine-grained pruning은 정확도 보존에는 유리할 수 있지만 불규칙 sparse pattern 때문에 일반 하드웨어에서 speedup이 제한된다.
- Channel/kernel/vector pruning은 구조적이라 hardware-friendly하지만 정보 손실이 커질 수 있다.
- Sensitivity analysis는 layer별 pruning 허용량을 찾는 실험이다.
- Global magnitude pruning은 전체 weight magnitude 기준으로 threshold를 잡는다.
- Iterative pruning은 pruning과 fine-tuning을 반복해 one-shot보다 안정적으로 정확도를 회복한다.
