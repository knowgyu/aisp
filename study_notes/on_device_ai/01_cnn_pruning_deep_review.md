# Chapter 1. Pruning for CNN

## 1. 이 챕터의 목적

CNN 모델에서 **덜 중요한 파라미터/구조를 제거(pruning)** 해서 모델을 가볍게 만들고, 그 과정에서 정확도와 효율성이 어떻게 바뀌는지 이해한다.

이 챕터의 큰 축은 3개다.

1. **Pruning Granularity**
   - 얼마나 작은/큰 단위로 제거할 것인가?
   - Fine-grained, vector-level, kernel-level, channel-level을 비교한다.
2. **Pruning Ratio**
   - 얼마나 많이 제거할 것인가?
   - Layer-wise 방식과 global 방식의 차이를 본다.
3. **Pruning Schedule**
   - 한 번에 제거할 것인가, 여러 번 나눠 제거할 것인가?
   - One-shot, iterative pruning, linear/cubic sparsity schedule을 본다.

시험 관점에서 이 챕터는 단순히 “작은 weight를 0으로 만든다”가 아니라,
**무엇을 기준으로 / 어떤 단위로 / 어느 정도 / 어떤 일정으로 제거하느냐**를 구분하는 게 핵심이다.

## 2. 핵심 용어

| 용어 | 의미 | 시험 포인트 |
|---|---|---|
| Pruning | 중요도가 낮은 weight 또는 구조를 제거하는 압축 기법 | 정확도-압축률 trade-off |
| Sparsity | 전체 파라미터 중 0 또는 제거된 비율 | 높을수록 압축은 크지만 성능 저하 위험 증가 |
| Dense model | pruning 전 원본 모델 | baseline accuracy/size 기준 |
| Fine-grained pruning | 개별 weight 단위 pruning | 높은 sparsity 가능, 하드웨어 가속은 어려울 수 있음 |
| Structured pruning | vector/kernel/channel처럼 구조 단위 pruning | 실제 속도 향상에 더 유리 |
| Reconstruction error | pruning 전후 출력 차이 | pruning이 layer 출력에 준 손상 측정 |
| Layer-wise pruning | layer마다 별도 ratio 적용 | 민감한 layer 보호 가능 |
| Global pruning | 전체 weight를 한 번에 비교해 pruning | 전역적으로 작은 weight 제거 |
| One-shot pruning | 목표 sparsity까지 한 번에 pruning | 간단하지만 성능 손상 위험 큼 |
| Iterative pruning | 조금씩 pruning하고 fine-tuning 반복 | 보통 안정적이나 시간 비용 증가 |

## 3. 핵심 수식

### 3.1 Sparsity

전체 파라미터 수가 `N`, 0이 아닌 파라미터 수가 `NNZ`라면:

```text
sparsity = 1 - NNZ / N
```

즉, sparsity가 0.9이면 전체 weight의 90%가 0이라는 뜻이다.

### 3.2 Reconstruction Error

입력 feature map을 `X`, 원래 weight를 `W`, pruning 후 weight를 `W_hat`이라고 하면:

```text
|| W X - W_hat X ||_2^2
```

의미:
- 원래 layer 출력: `W X`
- pruning 후 layer 출력: `W_hat X`
- 둘의 차이가 작으면 pruning을 해도 해당 layer의 동작이 크게 변하지 않았다는 뜻

## 4. 코드 셀별 해설

### Cell 0-1: 제목과 Goals

챕터 제목은 `Pruning CNN`이고, 실습 목표는 CNN 모델에 여러 pruning 기법을 적용해보는 것이다.

핵심은 다음 3단계 흐름이다.

```text
원본 VGG/CIFAR-10 모델 준비
→ pruning 적용
→ 모델 크기/정확도/reconstruction error 비교
```

여기서 비교 대상은 항상 dense model이다. 즉 pruning 전 모델의 정확도와 크기를 먼저 측정해야 이후 결과를 해석할 수 있다.

### Cell 3-5: Setup과 seed 설정

실습에서는 대략 다음 라이브러리를 사용한다.

- `torch`, `torch.nn`: 모델, layer, tensor 연산
- `torch.optim`, `LambdaLR`: fine-tuning과 pruning schedule
- `DataLoader`, `Dataset`: CIFAR-10 입력 파이프라인
- `torchvision.transforms`: image augmentation / tensor 변환
- `matplotlib`: weight distribution 시각화
- `numpy`, `random`: 재현성 및 수치 처리

Seed를 고정하는 이유:

```text
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
```

pruning 실험은 정확도 차이가 몇 %p 단위로 비교될 수 있어서, random crop, horizontal flip, weight 초기화, 학습 순서 등이 매번 달라지면 비교가 어려워진다.

### Cell 6-7: CIFAR-10 데이터셋 준비

CIFAR-10은 다음 특징을 가진다.

```text
이미지 크기: 32 x 32
채널: RGB 3채널
클래스 수: 10
전체: 60,000장
train: 50,000장
test: 10,000장
```

train transform:
- `RandomCrop(32, padding=4)`
- `RandomHorizontalFlip()`
- `ToTensor()`

의미:
- crop/flip은 data augmentation이다.
- 모델이 위치나 좌우 반전에 덜 민감하게 학습되도록 만든다.
- test에서는 augmentation 없이 `ToTensor()`만 적용한다.

DataLoader의 역할:

```text
Dataset에서 sample을 꺼냄
→ batch로 묶음
→ model 입력 형태 `[batch, 3, 32, 32]`로 전달
```

여기서 batch size는 512로 설정되어 있다.

### Cell 8-9: pretrained VGG 모델 로딩

실습은 CIFAR-10에 대해 이미 학습된 VGG 계열 모델을 불러와서 pruning한다.

중요한 이유:
- pruning은 보통 이미 성능이 좋은 dense model에서 시작한다.
- 처음부터 random model을 pruning하면 “중요한 weight”라는 개념이 거의 없다.
- 그래서 baseline model의 정확도와 크기가 먼저 필요하다.

모델은 GPU가 있으면 CUDA로 이동한다.

```text
if torch.cuda.is_available():
    model.cuda()
```

시험 포인트:
- pruning 대상은 보통 `Conv2d`, `Linear`의 weight이다.
- BatchNorm, bias, scalar parameter는 보통 별도로 다룬다.

### Cell 10-11: 모델 크기 평가

파라미터 수를 세는 함수가 나온다.

핵심 아이디어:

```text
전체 파라미터 수 = 모든 parameter.numel()의 합
nonzero 파라미터 수 = 모든 parameter.count_nonzero()의 합
모델 크기 = 파라미터 수 x bit-width / 8
```

여기서는 기본적으로 FP32를 가정하므로 `data_width = 32` bit이다.

```text
size(bytes) = num_parameters * 32 / 8
size(MiB) = size(bytes) / 1024^2
```

왜 `count_nonzero_only`가 필요한가?
- pruning 후 weight가 0이 되면 전체 `numel()`은 변하지 않는다.
- 하지만 실제로 sparse format으로 저장할 수 있다면 nonzero 개수가 압축 가능성을 나타낸다.
- 따라서 dense size와 effective sparse size를 비교하려면 nonzero count가 필요하다.

주의:
- 단순히 weight를 0으로 만든다고 PyTorch dense tensor의 메모리 사용량이 자동으로 줄지는 않는다.
- 실제 속도/메모리 이득은 sparse kernel 또는 structured pruning 여부에 달려 있다.

### Cell 12-14: 모델 정확도 평가

`evaluate()` 함수는 test DataLoader를 돌면서 accuracy를 계산한다.

흐름:

```text
model.eval()
for inputs, targets in dataloader:
    outputs = model(inputs)
    preds = outputs.argmax(dim=1)
    correct += (preds == targets).sum()
accuracy = correct / num_samples * 100
```

중요한 점:
- `model.eval()`은 BatchNorm/Dropout의 동작을 평가 모드로 바꾼다.
- `@torch.no_grad()`는 gradient 계산을 끄므로 평가가 빠르고 메모리를 덜 쓴다.
- CIFAR-10은 10-class classification이므로 `argmax(dim=1)`로 예측 class를 고른다.

시험 포인트:
- pruning 전 `dense_model_accuracy`가 기준선이다.
- 이후 pruning 결과는 “accuracy drop이 얼마나 생겼는가?”로 해석한다.

### Cell 15-16: 모델 가중치 분포 확인

weight histogram을 그려서 값의 분포를 본다.

왜 중요한가?
- magnitude-based pruning은 절댓값이 작은 weight를 덜 중요하다고 본다.
- weight가 0 근처에 많이 몰려 있으면 pruning 후보가 많다는 의미다.

코드의 핵심 조건:

```text
param.dim() > 1
```

의미:
- Conv2d weight shape: `[out_channels, in_channels, kernel_h, kernel_w]`
- Linear weight shape: `[out_features, in_features]`
- BatchNorm weight/bias: 보통 `[channels]`

따라서 `param.dim() > 1`은 Conv/Linear weight 위주로 보겠다는 뜻이다.

`count_nonzero_only=True`를 쓰면 pruning 후 0이 된 weight를 제외하고 남은 weight 분포만 볼 수 있다.

### Cell 18-19: Reconstruction Error

Reconstruction error는 pruning 전후의 layer 출력 차이를 측정한다.

직관:

```text
원래 layer: y = W x
pruned layer: y_hat = W_hat x
error = || y - y_hat ||^2
```

이 값이 작으면:
- pruning 후에도 해당 layer가 비슷한 출력을 낸다.
- 즉, 제거한 weight들이 layer 동작에 상대적으로 덜 중요했을 가능성이 높다.

이 값이 크면:
- pruning이 layer 표현을 크게 망가뜨렸다는 뜻이다.
- accuracy drop으로 이어질 가능성이 커진다.

## 5. 실습 빈칸 의도와 정답 비교

> 아직 구현부 진입 전. Fine-grained pruning 실습부터 기록 예정.

## 6. 실험 결과 해석

현재까지 기준선으로 봐야 할 값:

- dense model size
- dense model accuracy
- dense weight distribution
- reconstruction error 계산 방식

이 4개가 이후 모든 pruning 실험의 비교 기준이다.

## 7. 시험 예상 질문

### Q1. Pruning에서 sparsity란 무엇인가?

전체 파라미터 중 제거되었거나 0이 된 파라미터의 비율이다. `1 - nonzero / total`로 계산한다.

### Q2. Fine-grained pruning이 실제 속도 향상으로 바로 이어지지 않을 수 있는 이유는?

개별 weight만 0이 되어도 tensor shape은 그대로라 dense 연산을 쓰면 계산량이 그대로일 수 있다. 실제 이득을 얻으려면 sparse kernel이나 구조적 pruning이 필요하다.

### Q3. Pruning 전 dense model accuracy를 먼저 측정하는 이유는?

pruning 후 정확도 저하를 해석할 baseline이 필요하기 때문이다.

### Q4. `param.dim() > 1` 조건으로 weight를 보는 이유는?

Conv2d/Linear weight는 2차원 이상이고, BatchNorm이나 bias처럼 1차원 parameter는 pruning 대상에서 제외하려는 의도다.

### Q5. Reconstruction error는 무엇을 측정하는가?

pruning 전 layer 출력과 pruning 후 layer 출력의 차이를 측정한다. 작을수록 pruning 후 layer 동작이 덜 변했다는 뜻이다.

## 8. 내가 헷갈린 점

- “0으로 만든 weight 수가 많다”와 “실제 모델 파일/추론이 빨라진다”는 같은 말이 아니다.
- unstructured sparsity는 압축 가능성을 보여주지만, 하드웨어 효율은 별도 문제다.
- pruning의 기준은 accuracy 하나만이 아니라 model size, FLOPs, reconstruction error 등 여러 지표로 봐야 한다.

---

## 9. 1.1 Pruning Granularity / Pattern

### 9.1 이 섹션의 핵심

Pruning granularity는 **어떤 단위로 weight를 제거할 것인가**를 뜻한다.

같은 sparsity 50%라도 제거 단위가 다르면 결과가 달라진다.

```text
Fine-grained  <  Vector-level  <  Kernel-level  <  Channel-level
작은 단위                         큰 구조 단위
유연함 큼                         하드웨어 효율 유리
```

중요한 trade-off:

| 방식 | 제거 단위 | 장점 | 단점 |
|---|---|---|---|
| Fine-grained | 개별 weight | 정확도 보존에 유리, 가장 유연함 | 불규칙 sparsity라 실제 가속 어려움 |
| Vector-level | kernel 내부의 vector | fine-grained보다 구조적 | 구조 제약 때문에 손상 증가 가능 |
| Kernel-level | 하나의 2D conv kernel | 연산 구조와 더 잘 맞음 | 더 큰 단위 제거라 정확도 손상 가능 |
| Channel-level | 입력/출력 channel 전체 | 실제 모델 slimming/가속에 가장 유리 | 가장 거친 단위라 성능 손상 위험 큼 |

### 9.2 실습 대상 weight shape

실습에서는 `model.backbone.conv1.weight` 같은 convolution weight를 대상으로 한다.

Conv2d weight shape은 보통 다음과 같다.

```text
[out_channels, in_channels, kernel_h, kernel_w]
```

예를 들어 실습 reconstruction cell에서는 다음 shape을 가정한다.

```text
input_sample: [1, 64, 32, 32]
conv weight:  [128, 64, 3, 3]
output:       [1, 128, 32, 32]
```

여기서 의미:

- `out_channels = 128`: 출력 feature map 채널 수
- `in_channels = 64`: 입력 feature map 채널 수
- `kernel_h = 3`, `kernel_w = 3`: 각 convolution kernel 크기

즉 weight 하나는 다음처럼 볼 수 있다.

```text
weight[o, i, :, :] = 입력 채널 i에서 출력 채널 o로 연결되는 3x3 kernel
```

### 9.3 Fine-grained pruning

#### 개념

Fine-grained pruning은 weight tensor 안의 **개별 scalar weight**를 기준으로 제거한다.

중요도는 보통 절댓값이다.

```text
importance = abs(weight)
```

작은 절댓값을 가진 weight는 출력에 미치는 영향이 작다고 가정하고 0으로 만든다.

#### 코드 의도

빈칸의 핵심 로직은 다음이다.

```python
num_pruned_elements = round(weight.numel() * sparsity)
importance = torch.abs(weight)
threshold = torch.kthvalue(importance.flatten(), num_pruned_elements).values
mask = importance > threshold
weight.mul_(mask)
```

단, 실제 구현에서는 edge case가 중요하다.

```text
sparsity = 0.0 → 전부 유지 → mask all ones
sparsity = 1.0 → 전부 제거 → mask all zeros
```

#### 왜 kthvalue를 쓰나?

`sparsity = 0.5`이면 전체 weight 중 절댓값이 작은 50%를 제거해야 한다.

`torch.kthvalue(x, k)`는 정렬했을 때 k번째 작은 값을 찾는다.
따라서 threshold를 구한 뒤 threshold보다 큰 값만 살리면 작은 값들이 제거된다.

#### 시험 포인트

Fine-grained pruning은 같은 sparsity에서 보통 reconstruction error가 작을 가능성이 높다.
왜냐하면 가장 덜 중요한 개별 weight만 골라 제거할 수 있기 때문이다.
하지만 0이 흩어져 있으므로 dense hardware에서는 속도 향상이 바로 나오지 않을 수 있다.

### 9.4 Vector-level pruning

#### 개념

Vector-level pruning은 개별 weight가 아니라 convolution kernel 내부의 **1D vector 단위**를 제거한다.

Conv weight가 `[out, in, h, w]`일 때, 실습 코드 흐름상 vector importance는 대략 다음처럼 계산된다.

```python
importance = weight.abs().sum(dim=3, keepdim=True)
```

즉 마지막 차원 `kernel_w` 방향으로 묶어서 하나의 vector 중요도를 계산한다.

예시:

```text
weight[o, i, h, :]  # 길이 kernel_w인 vector
```

이 vector 전체의 L1 norm이 작으면 그 vector 전체를 제거한다.

#### 코드 의도

```python
num_vectors = out_channels * in_channels * kernel_h
num_pruned_vectors = round(num_vectors * sparsity)
importance = weight.abs().sum(dim=3, keepdim=True)
threshold = torch.kthvalue(importance.flatten(), num_pruned_vectors).values
mask = importance > threshold
mask = mask.expand_as(weight)
weight.mul_(mask)
```

중요한 점은 `mask.expand_as(weight)`다.

- importance shape: `[out, in, h, 1]`
- weight shape: `[out, in, h, w]`
- 마지막 차원의 vector 전체에 같은 mask를 broadcast한다.

#### 시험 포인트

Vector-level은 fine-grained보다 구조적이다.
하지만 vector 안의 개별 weight 중 일부가 중요해도 vector 전체가 제거될 수 있다.
그래서 fine-grained보다 정확도 손상 가능성이 커진다.

### 9.5 Kernel-level pruning

#### 개념

Kernel-level pruning은 `weight[o, i, :, :]` 하나, 즉 **입력 채널 i에서 출력 채널 o로 가는 2D kernel 전체**를 제거한다.

중요도는 kernel 전체의 L1 norm이다.

```python
importance = weight.abs().sum(dim=(2, 3), keepdim=True)
```

shape 관점:

```text
weight:     [out, in, h, w]
importance: [out, in, 1, 1]
mask:       [out, in, h, w]로 expand
```

#### 코드 의도

```python
num_kernels = out_channels * in_channels
num_pruned_kernels = round(num_kernels * sparsity)
importance = weight.abs().sum(dim=(2, 3), keepdim=True)
threshold = torch.kthvalue(importance.flatten(), num_pruned_kernels).values
mask = importance > threshold
mask = mask.expand_as(weight)
weight.mul_(mask)
```

#### 시험 포인트

Kernel-level은 fine/vector보다 더 coarse하다.
하나의 3x3 kernel이 통째로 0이 되므로 구조적 sparsity가 강해진다.
하지만 중요한 weight 몇 개가 포함되어 있어도 kernel norm이 작으면 전체가 제거된다.

### 9.6 Channel-level pruning

#### 개념

Channel-level pruning은 입력 또는 출력 channel 전체를 제거한다.
실습 코드 흐름상 입력 channel 기준으로 중요도를 계산한다.

```python
importance = weight.abs().sum(dim=(0, 2, 3), keepdim=True)
```

shape 의미:

```text
weight: [out, in, h, w]
입력 channel i의 중요도 = 모든 output channel과 kernel 위치에 걸친 abs sum
```

즉 `in_channels` 중 어떤 입력 채널 전체가 덜 중요한지 본다.

#### 코드 의도

```python
num_channels = weight.shape[1]
num_pruned_channels = round(num_channels * sparsity)
importance = weight.abs().sum(dim=(0, 2, 3), keepdim=True)
threshold = torch.kthvalue(importance.flatten(), num_pruned_channels).values
mask = importance > threshold
mask = mask.expand_as(weight)
weight.mul_(mask)
```

#### 시험 포인트

Channel-level은 가장 구조적인 pruning이다.
실제로 channel을 제거하면 다음 layer의 입력 채널도 같이 줄일 수 있어서 모델 slimming과 실제 가속에 유리하다.
하지만 제거 단위가 너무 크므로 정확도 손상 위험도 가장 크다.

### 9.7 Importance Sum 비교

실습에서는 각 pruning 방식 후 남은 weight들의 importance sum을 비교한다.

```python
importance = weight.abs()
importance_sum_original = importance.sum()
importance_sum_pruned = (importance * mask).sum()
```

해석:

- 남은 importance sum이 높다 = 중요한 weight를 많이 보존했다.
- 남은 importance sum이 낮다 = 중요한 weight까지 많이 제거했을 수 있다.

보통 예상 순서:

```text
Original > Fine-grained ≥ Vector-level ≥ Kernel-level ≥ Channel-level
```

정확히 항상 이 순서라고 단정하면 안 되지만, 작은 단위 pruning일수록 중요도가 낮은 것만 골라 제거하기 쉬워서 남은 importance sum이 커지는 경향이 있다.

### 9.8 Reconstruction Error 비교

각 pruning mask를 적용한 conv layer를 만들어 같은 input에 대해 출력 차이를 비교한다.

흐름:

```text
1. random input_sample 생성
2. original conv output 계산
3. fine/vector/kernel/channel pruned conv output 계산
4. original output과 pruned output의 L2 error 비교
```

해석:

| reconstruction error | 의미 |
|---|---|
| 작음 | 원래 layer 출력과 비슷함, pruning 손상이 작음 |
| 큼 | layer 출력이 많이 변함, accuracy drop 위험 큼 |

예상 경향:

```text
Fine-grained error가 가장 작고,
Channel-level error가 가장 클 가능성이 높다.
```

이유:
- Fine-grained는 정말 작은 scalar weight만 골라 제거 가능하다.
- Channel-level은 채널 전체를 날리므로 출력 feature map에 미치는 영향이 크다.

### 9.9 이 섹션 한 줄 요약

Pruning granularity는 **정확도 보존과 실제 하드웨어 효율 사이의 trade-off**다.
Fine-grained는 정확도에 유리하지만 불규칙하고, channel-level은 실제 가속에 유리하지만 성능 손상이 클 수 있다.

## 10. 1.1 시험 예상 질문 추가

### Q6. Fine-grained pruning과 channel-level pruning의 가장 큰 차이는?

Fine-grained는 개별 scalar weight를 제거하고, channel-level은 입력 또는 출력 channel 전체를 제거한다. Fine-grained는 유연해서 정확도 보존에 유리하지만 실제 가속은 어려울 수 있고, channel-level은 구조적으로 제거되므로 실제 모델 크기/연산량 감소에 유리하지만 정확도 손상 위험이 크다.

### Q7. Kernel-level pruning에서 importance를 어떻게 계산하는가?

각 2D kernel의 weight 절댓값 합, 즉 L1 norm으로 계산한다. Conv weight가 `[out, in, h, w]`라면 보통 `sum(dim=(2,3))`으로 각 `[out, in]` kernel의 중요도를 구한다.

### Q8. `mask.expand_as(weight)`는 왜 필요한가?

vector/kernel/channel 단위로 만든 mask는 weight보다 작은 shape을 가진다. 해당 구조 단위 전체에 같은 mask를 적용하려면 원래 weight shape으로 broadcast/expand해야 한다.

### Q9. Reconstruction error가 accuracy와 완전히 같은 지표인가?

아니다. Reconstruction error는 특정 layer 출력 변화량이고, accuracy는 전체 모델의 최종 분류 성능이다. 하지만 reconstruction error가 크면 layer 표현이 많이 바뀐 것이므로 accuracy drop 위험이 커질 수 있다.

### Q10. 같은 50% sparsity라도 pruning 방식에 따라 결과가 다른 이유는?

제거 단위가 다르기 때문이다. Fine-grained는 작은 weight 50%를 개별적으로 고를 수 있지만, channel-level은 채널 전체 단위로 제거해야 하므로 중요한 weight도 함께 제거될 수 있다.

---

## 11. 1.2 Pruning Ratio

### 11.1 이 섹션의 핵심

Pruning granularity가 “무엇을 자를 것인가?”였다면, pruning ratio는 **얼마나 자를 것인가?**이다.

여기서 중요한 사실:

```text
모든 layer가 pruning에 똑같이 강하지 않다.
```

어떤 layer는 sparsity를 높여도 accuracy가 잘 버티고, 어떤 layer는 조금만 잘라도 accuracy가 크게 떨어진다.
그래서 이 섹션은 layer별 sensitivity를 먼저 분석한 뒤, pruning ratio를 다르게 주는 방법을 배운다.

### 11.2 Uniform vs Non-uniform pruning

| 방식 | 의미 | 장점 | 단점 |
|---|---|---|---|
| Uniform pruning | 모든 layer에 같은 sparsity 적용 | 단순함 | 민감한 layer도 똑같이 잘려 성능 저하 가능 |
| Non-uniform pruning | layer별로 다른 sparsity 적용 | 민감한 layer 보호 가능 | sensitivity 분석 필요 |
| Global pruning | 전체 layer weight를 한 곳에 모아 threshold 결정 | 전역적으로 작은 weight 제거 | layer별 sparsity가 자동으로 달라짐 |

시험에서 자주 나오는 구분:

```text
Uniform = layer마다 같은 비율
Global = 전체 weight를 한 번에 비교
Non-uniform = layer별 pruning ratio가 다름
```

Global pruning은 결과적으로 non-uniform sparsity를 만들 수 있다.
왜냐하면 어떤 layer는 작은 weight가 많고, 어떤 layer는 큰 weight가 많기 때문이다.

### 11.3 Sensitivity analysis

#### 목적

특정 layer 하나만 pruning하고 나머지 layer는 원래대로 둔 상태에서 accuracy 변화를 본다.

흐름:

```text
for each layer:
    원본 weight 백업
    for sparsity in [0.4, 0.5, ..., 0.9]:
        해당 layer만 pruning
        model accuracy 평가
        원본 weight 복구
```

이렇게 하면 “이 layer는 pruning에 민감한가?”를 알 수 있다.

#### 코드 의도

`sensitivity_scan()`의 핵심은 다음이다.

```python
for name, param in model.named_parameters():
    if param.dim() > 1:
        param_clone = param.detach().clone()
        for sparsity in sparsities:
            prune_weight_fine_grained(param.detach(), sparsity)
            acc = evaluate(model, dataloader)
            accuracy.append(acc)
            param.copy_(param_clone)
```

중요한 점:

1. `param.dim() > 1`
   - Conv/Linear weight만 대상으로 한다.
2. `param_clone = param.detach().clone()`
   - pruning 후 원래 weight로 되돌리기 위해 백업한다.
3. layer 하나씩만 pruning
   - 여러 layer를 동시에 pruning하면 어떤 layer가 원인인지 모른다.
4. `@torch.no_grad()`
   - sensitivity 측정은 학습이 아니라 평가이므로 gradient가 필요 없다.

### 11.4 Sensitivity curve 해석

그래프는 보통 x축이 sparsity, y축이 accuracy다.

```text
x축: sparsity
 y축: top-1 accuracy
```

해석:

| 곡선 모양 | 의미 |
|---|---|
| sparsity가 올라가도 accuracy가 천천히 떨어짐 | pruning에 둔감한 layer |
| sparsity 조금만 올라가도 accuracy 급락 | pruning에 민감한 layer |
| 0.8 이상에서 갑자기 떨어짐 | 어느 지점까지는 버티지만 threshold 이후 손상 큼 |

실습에서는 dense model accuracy의 96%를 lower bound로 둔다.

```text
lower_bound_accuracy = dense_model_accuracy * 0.96
```

의미:
- 원본 정확도의 96% 이상은 유지하자는 기준이다.
- 예: dense accuracy가 90%라면 lower bound는 86.4%.

이 기준선을 넘지 않는 범위에서 layer별 sparsity를 선택한다.

### 11.5 실습 2: Sensitivity 기반 custom sparsity

실습에서는 sensitivity curve를 보고 `custom_sparsity_dict`를 채운다.

예상 형태:

```python
custom_sparsity_dict = {
    'backbone.conv0.weight': ...,
    'backbone.conv1.weight': ...,
    'backbone.conv2.weight': ...,
    'backbone.conv3.weight': ...,
    'backbone.conv4.weight': ...,
    'backbone.conv5.weight': ...,
    'backbone.conv6.weight': ...,
    'backbone.conv7.weight': ...,
    'classifier.weight': ...,
}
```

선택 원칙:

```text
민감한 layer → 낮은 sparsity
둔감한 layer → 높은 sparsity
```

특히 일반적으로 CNN에서는 다음 경향을 자주 조심한다.

- 첫 layer는 입력 이미지의 low-level feature를 받으므로 너무 세게 자르면 위험할 수 있다.
- 마지막 classifier는 class decision에 직접 연결되므로 민감할 수 있다.
- 중간 convolution layer는 redundancy가 커서 상대적으로 더 pruning 가능할 수 있다.

단, 이건 일반 경향이고 실제 선택은 sensitivity curve를 기준으로 해야 한다.

### 11.6 FineGrainedPruner 클래스의 역할

`FineGrainedPruner`는 layer별 sparsity dict를 받아서 mask를 만들고 적용한다.

역할:

```text
1. 각 layer별 pruning mask 생성
2. model parameter에 mask 적용
3. 필요하면 원본 weight 복구
```

중요한 구조:

```text
self.masks[name] = mask
```

이렇게 layer 이름별로 mask를 저장한다.

왜 mask를 저장해야 하나?
- pruning 후 fine-tuning할 때, 0으로 만든 weight가 gradient update로 다시 살아날 수 있다.
- mask를 callback 등으로 반복 적용하면 pruned weight를 계속 0으로 유지할 수 있다.

### 11.7 모델 sparsity 계산

실습에서는 pruning 후 전체 model sparsity를 계산한다.

```python
num_nonzeros = sum(param.count_nonzero())
num_elements = sum(param.numel())
sparsity = 1 - num_nonzeros / num_elements
```

주의:
- layer별 sparsity를 다르게 줬기 때문에 전체 model sparsity는 단순 평균이 아니다.
- parameter 수가 많은 layer의 sparsity가 전체 sparsity에 더 큰 영향을 준다.

### 11.8 Uniform sparsity와 비교

Sensitivity 기반 custom pruning 결과를 uniform pruning과 비교한다.

비교 방식:

```text
uniform sparsity 0.4, 0.5, ..., 0.9 각각 적용
→ accuracy 측정
→ custom pruning의 sparsity/accuracy 점과 비교
```

해석:

좋은 custom pruning이면:

```text
같은 sparsity에서 uniform보다 accuracy가 높거나,
같은 accuracy에서 uniform보다 sparsity가 높다.
```

즉, sensitivity analysis의 목적은 무작정 많이 자르는 게 아니라,
**성능 손상이 작은 곳을 더 많이 자르고 민감한 곳은 보호하는 것**이다.

### 11.9 실습 3: Global Magnitude Pruning

Global magnitude pruning은 layer별로 따로 threshold를 구하지 않는다.
모든 pruning 대상 weight를 하나의 긴 vector로 모은 뒤, 전역 threshold를 구한다.

#### 코드 의도

빈칸의 핵심은 다음이다.

```python
parameters_to_prune = []
for name, param in model.named_parameters():
    if param.dim() > 1:
        parameters_to_prune.append(param.view(-1))

all_weights = torch.cat(parameters_to_prune)
num_elements = all_weights.numel()
num_zeros = round(num_elements * sparsity)
threshold = torch.kthvalue(torch.abs(all_weights), num_zeros).values
```

그 다음 각 layer에 대해:

```python
mask = torch.abs(param.data) > threshold
```

을 적용한다.

#### Global pruning의 의미

Global pruning은 전체 모델에서 절댓값이 작은 weight를 제거한다.
따라서 layer별 sparsity는 자동으로 달라진다.

예:

```text
Layer A: 작은 weight가 많음 → 많이 pruning됨
Layer B: 큰 weight가 많음 → 적게 pruning됨
```

이 점 때문에 global pruning은 uniform pruning보다 효율적일 수 있다.

### 11.10 Layer-wise vs Global 비교

| 항목 | Layer-wise / Uniform | Sensitivity custom | Global magnitude |
|---|---|---|---|
| threshold | layer마다 별도 또는 같은 ratio | 사람이 curve 보고 지정 | 전체 weight 기준 하나 |
| layer별 sparsity | 같거나 수동 지정 | 수동 지정 | 자동으로 달라짐 |
| 장점 | 단순함 | 민감한 layer 보호 | 전역적으로 작은 weight 제거 |
| 단점 | 비효율 가능 | 분석 필요 | layer 구조/민감도 직접 고려는 안 함 |

시험에서 중요한 문장:

> Global pruning은 전체 weight distribution을 기준으로 pruning하므로, 각 layer의 sparsity가 같지 않을 수 있다.

## 12. 1.2 시험 예상 질문 추가

### Q11. Sensitivity analysis는 왜 필요한가?

모든 layer가 pruning에 같은 정도로 강하지 않기 때문이다. 특정 layer만 pruning하면서 accuracy 변화를 측정하면 어떤 layer를 많이 잘라도 되는지, 어떤 layer를 보호해야 하는지 알 수 있다.

### Q12. Uniform pruning의 단점은?

모든 layer에 같은 sparsity를 적용하므로 민감한 layer까지 과하게 pruning될 수 있다. 이 때문에 같은 전체 sparsity에서도 accuracy가 크게 떨어질 수 있다.

### Q13. Global magnitude pruning은 layer-wise pruning과 무엇이 다른가?

Layer-wise pruning은 각 layer 안에서 threshold를 정하지만, global pruning은 모든 layer의 weight를 하나로 모아 전역 threshold를 정한다. 따라서 layer별 sparsity가 자동으로 달라질 수 있다.

### Q14. `custom_sparsity_dict`를 어떻게 정해야 하는가?

Sensitivity curve를 보고 accuracy가 기준선 아래로 떨어지지 않는 범위에서 layer별 sparsity를 정한다. 민감한 layer는 낮게, 둔감한 layer는 높게 설정한다.

### Q15. Pruning 후 mask를 계속 적용해야 하는 이유는?

Fine-tuning 중 gradient update로 0이 된 weight가 다시 nonzero가 될 수 있기 때문이다. mask를 반복 적용하면 pruned weight를 계속 0으로 유지할 수 있다.

---

## 13. 1.3 Pruning Schedule

### 13.1 이 섹션의 핵심

Pruning schedule은 **언제, 얼마나 빠르게 sparsity를 올릴 것인가**를 정한다.

앞에서는 pruning ratio를 정했다. 이제는 그 ratio까지 도달하는 방법을 비교한다.

```text
One-shot pruning: 목표 sparsity를 한 번에 적용
Iterative pruning: sparsity를 조금씩 올리며 pruning + fine-tuning 반복
```

핵심 trade-off:

| 방식 | 장점 | 단점 |
|---|---|---|
| One-shot | 단순하고 빠름 | 모델이 갑자기 손상되어 accuracy drop이 클 수 있음 |
| Iterative | 모델이 적응할 시간을 줌 | 학습 시간이 더 오래 걸림 |

### 13.2 One-shot pruning

One-shot pruning은 목표 sparsity를 한 번에 적용한다.

실습에서는 대략 다음 설정을 사용한다.

```text
목표 sparsity: 0.95
fine-tuning epoch: 5
optimizer: SGD
learning rate: 0.01
momentum: 0.9
weight_decay: 1e-4
scheduler: CosineAnnealingLR
loss: CrossEntropyLoss
```

흐름:

```python
pruner = FineGrainedPrunerV2(model, target_sparsity, global_prune=True)
for epoch in range(num_finetune_epochs):
    train(..., callbacks=[lambda: pruner.apply(model)])
    accuracy = evaluate(model, dataloader['test'])
```

중요한 부분은 callback이다.

```python
callbacks=[lambda: pruner.apply(model)]
```

왜 필요한가?

- pruning mask로 0이 된 weight도 optimizer step 이후 다시 nonzero가 될 수 있다.
- 매 training step 이후 mask를 다시 곱하면 pruned weight가 계속 0으로 유지된다.

즉, pruning 상태를 유지하면서 남은 weight만 fine-tuning하는 구조다.

### 13.3 Iterative pruning

Iterative pruning은 처음부터 95%를 자르지 않고, epoch이 진행될수록 sparsity를 조금씩 높인다.

예:

```text
epoch 0: sparsity 0.90
epoch 1: sparsity 0.925
epoch 2: sparsity 0.95
이후: 0.95 유지
```

의미:

- 처음에는 덜 자른 상태로 모델이 적응한다.
- 그 다음 조금 더 자른다.
- 목표 sparsity에 도달한 뒤 fine-tuning을 계속한다.

이 방식은 one-shot보다 안정적일 수 있다.

### 13.4 Linear sparsity scheduler

Linear scheduler는 sparsity를 일정한 속도로 증가시킨다.

수식:

```text
v(t) = v_f + (v_i - v_f) * (1 - (t - t_i) / (t_f - t_i))
```

동일하게 쓰면 더 직관적으로는:

```text
progress = (t - t_i) / (t_f - t_i)
v(t) = v_i + (v_f - v_i) * progress
```

예:

```text
num_epochs = 5
start epoch = 0
end epoch = 2
sparsity_start = 0.90
sparsity_end = 0.95
```

결과:

```text
[0.90, 0.925, 0.95, 0.95, 0.95]
```

해석:

- epoch 0: 90%
- epoch 1: 92.5%
- epoch 2: 95%
- epoch 3-4: 목표 sparsity 유지

### 13.5 Cubic sparsity scheduler

Cubic scheduler는 세제곱 형태로 sparsity를 증가시킨다.

수식:

```text
v(t) = v_f + (v_i - v_f) * (1 - (t - t_i) / (t_f - t_i))^3
```

예시 결과:

```text
[0.90, 0.94375, 0.95, 0.95, 0.95]
```

여기서 중요한 직관:

- linear는 매 epoch 일정하게 증가한다.
- cubic은 증가 속도가 일정하지 않다.
- pruning schedule의 모양에 따라 모델이 적응하는 방식이 달라질 수 있다.

강의 노트 문맥에서는 cubic scheduler가 linear보다 일반적으로 더 좋은 성능을 보일 수 있다고 설명한다.
핵심 이유는 pruning 변화가 더 부드럽거나 적응 시간을 더 주는 형태가 될 수 있기 때문이다.

### 13.6 `get_sparsity_schedule` 구현 의도

함수는 전체 epoch 수와 시작/끝 epoch, 시작/끝 sparsity, exponent를 받아 epoch별 sparsity list를 만든다.

의도는 다음과 같다.

```python
for epoch in range(num_epochs):
    if epoch < epoch_start:
        sparsity = sparsity_start
    elif epoch > epoch_end:
        sparsity = sparsity_end
    else:
        progress = (epoch - epoch_start) / (epoch_end - epoch_start)
        sparsity = sparsity_end + (sparsity_start - sparsity_end) * (1 - progress) ** exponent
    sparsity_schedule.append(sparsity)
```

`exponent = 1`이면 linear, `exponent = 3`이면 cubic이다.

주의할 점:
- `epoch_end == epoch_start`일 때 division by zero를 조심해야 한다.
- schedule은 epoch 단위이고, optimizer scheduler는 batch step 단위일 수 있다.

### 13.7 Iterative pruning 코드 흐름

Linear schedule 사용:

```python
for epoch in range(num_finetune_epochs):
    pruner = FineGrainedPrunerV2(model, linear_sparsity_schedule[epoch], global_prune=True)
    train(..., callbacks=[lambda: pruner.apply(model)])
    accuracy = evaluate(...)
```

Cubic schedule도 동일하고 sparsity schedule만 바뀐다.

중요한 점:

- 매 epoch마다 새 sparsity로 pruner를 만든다.
- 그 epoch의 목표 sparsity에 맞는 mask를 적용한다.
- training 중 callback으로 mask 유지한다.
- epoch 끝마다 accuracy를 본다.

### 13.8 One-shot vs Iterative 비교

| 항목 | One-shot | Iterative |
|---|---|---|
| pruning 적용 | 목표 sparsity 한 번에 적용 | epoch마다 점진적으로 증가 |
| 구현 | 단순 | schedule 필요 |
| 모델 충격 | 큼 | 상대적으로 작음 |
| 학습 시간 | 짧음 | 더 김 |
| accuracy 회복 | 어려울 수 있음 | 더 안정적일 수 있음 |

시험 답변용 한 문장:

> Iterative pruning은 sparsity를 점진적으로 증가시키며 fine-tuning을 반복하기 때문에, 모델이 pruning으로 인한 손상에 적응할 시간을 제공한다.

## 14. Chapter 1 통합 비교

### 14.1 전체 흐름

```text
1. Dense CNN 모델 준비
2. 모델 크기 / 정확도 / weight distribution 확인
3. pruning granularity 비교
4. pruning ratio 선택
5. sensitivity analysis로 layer별 민감도 확인
6. uniform / custom / global magnitude pruning 비교
7. one-shot / iterative pruning schedule 비교
8. fine-tuning으로 accuracy 회복 확인
```

### 14.2 세 가지 큰 축

| 축 | 질문 | 대표 개념 |
|---|---|---|
| Granularity | 무엇을 자를까? | fine/vector/kernel/channel |
| Ratio | 얼마나 자를까? | sparsity, sensitivity, global threshold |
| Schedule | 언제 어떻게 자를까? | one-shot, iterative, linear/cubic |

### 14.3 제일 중요한 시험 비교표

| 비교 | A | B | 핵심 차이 |
|---|---|---|---|
| Fine-grained vs Channel-level | 개별 weight | channel 전체 | 정확도 보존 vs 하드웨어 효율 |
| Uniform vs Global | layer마다 같은 비율 | 전체 weight 기준 threshold | layer별 sparsity 같음 vs 달라질 수 있음 |
| One-shot vs Iterative | 한 번에 pruning | 점진적 pruning | 빠름 vs 안정적 |
| Magnitude pruning vs Sensitivity-based | weight 크기 기준 | accuracy 민감도 고려 | 단순 기준 vs 실험 기반 ratio |

## 15. Chapter 1 최종 시험 예상 질문

### Q16. Pruning의 세 가지 설계 축은?

Granularity, ratio, schedule이다. Granularity는 어떤 단위로 자를지, ratio는 얼마나 자를지, schedule은 언제 어떻게 sparsity를 증가시킬지를 의미한다.

### Q17. Iterative pruning이 one-shot pruning보다 나을 수 있는 이유는?

한 번에 많은 weight를 제거하지 않고 점진적으로 제거하면서 fine-tuning하기 때문에, 모델이 손상에 적응할 시간이 생긴다.

### Q18. Global magnitude pruning이 non-uniform sparsity를 만드는 이유는?

전체 layer의 weight를 하나로 모아 전역 threshold를 정하기 때문에, 작은 weight가 많은 layer는 더 많이 잘리고 큰 weight가 많은 layer는 덜 잘린다.

### Q19. Pruning 후 fine-tuning에서 mask callback을 쓰는 이유는?

optimizer step으로 pruned weight가 다시 nonzero가 되는 것을 막기 위해서다. mask를 계속 적용해 제거된 weight를 0으로 유지한다.

### Q20. 모델 sparsity가 높아졌는데 실제 inference가 빨라지지 않을 수 있는 이유는?

weight 값만 0이 되었을 뿐 tensor shape과 dense 연산은 그대로일 수 있기 때문이다. 실제 가속을 위해서는 sparse kernel, structured pruning, channel 제거 후 모델 구조 변경 등이 필요하다.

## 16. Chapter 1 암기용 초압축

```text
Pruning = 덜 중요한 weight/구조 제거
Sparsity = 제거 비율
Magnitude pruning = abs(weight)가 작은 것 제거
Granularity = fine / vector / kernel / channel
Fine-grained = 정확도 유리, 가속 어려움
Channel-level = 가속 유리, 정확도 손상 위험
Sensitivity = layer 하나씩 잘라 accuracy 변화 보기
Uniform = 모든 layer 같은 비율
Global = 전체 weight 기준 threshold
One-shot = 한 번에 목표 sparsity
Iterative = 조금씩 pruning + fine-tuning
Mask callback = pruned weight가 다시 살아나는 것 방지
```
