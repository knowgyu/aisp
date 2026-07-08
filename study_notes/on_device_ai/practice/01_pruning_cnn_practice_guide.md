# On-Device AI Practice 01 — Pruning for CNN 셀별 코드 학습 가이드

> 오른쪽에는 원본 노트북 HTML을 띄우고, 왼쪽은 **모든 셀을 따라가는 해설 지도**로 쓴다. 이 문서는 원본 코드를 대체하지 않고, 각 셀이 왜 필요한지/무슨 shape로 움직이는지/직접 구현할 때 어디를 봐야 하는지 설명한다.

- 기준 교안: `ODAI-1 Chapter 2 Network Pruning`
- 원본 노트북: `On-Device AI 강의자료/실습/1. Pruning for CNN.ipynb`
- 학습 목표: VGG/CIFAR-10 모델에서 pruning granularity, pruning ratio, sensitivity, pruning schedule을 코드로 연결한다.

## 0. 그림으로 먼저 잡기

```mermaid
flowchart LR
  A["Dense VGG"]
  B["Weight 분포/기준 측정"]
  A --> B
  C["Granularity별 mask"]
  B --> C
  D["Sensitivity scan"]
  C --> D
  E["Layer-wise vs global"]
  D --> E
  F["Schedule + fine-tuning"]
  E --> F
```

### 실습 전체에서 계속 붙잡을 수식

$W_{pruned}=W\odot M,\; M\in\{0,1\}^{shape(W)}$

### 핵심 shape 표

| 대상 | Shape / 표현 | 의미 |
|---|---|---|
| CIFAR-10 batch | `[B, 3, 32, 32]` | RGB 이미지 mini-batch |
| Conv weight | `[C_out, C_in, K_h, K_w]` | pruning granularity가 정의되는 기본 텐서 |
| mask | `same as weight` | `1`은 유지, `0`은 제거 |
| logits | `[B, 10]` | CIFAR-10 class score |

## 1. 노트북 전체 지도

| 구간 | 셀 범위 | 오른쪽에서 보이는 제목 | 여기서 잡아야 할 것 |
|---:|---:|---|---|
| 1 | 001-001 | Assignment 1. Pruning for CNN | 프루닝 |
| 2 | 002-002 | Goals | 희소도, 프루닝, fine-grained, vector-level |
| 3 | 003-006 | Setup | 재현성, CIFAR-10 데이터, DataLoader, VGG 모델 |
| 4 | 007-008 | CIFAR-10 데이터셋 준비 | CIFAR-10 데이터, DataLoader |
| 5 | 009-010 | 모델 로딩 | CIFAR-10 데이터, VGG 모델, 프루닝 |
| 6 | 011-012 | 모델 크기 평가 | 모델 크기, zero-point |
| 7 | 013-015 | 모델의 정확도 평가 | DataLoader, 평가 루프, 학습 루프, 스케줄링 |
| 8 | 016-018 | 모델 가중치 값의 분포 확인 | 모델 크기, 프루닝, kernel-level, channel-level |
| 9 | 019-020 | Reconstruction Error 계산 함수 정의 | 프루닝 |
| 10 | 021-021 | 1.1. Pruning Granularity/Pattern | 프루닝 |
| 11 | 022-022 | Pruning Granularity 소개 | 프루닝, fine-grained, vector-level, kernel-level |
| 12 | 023-025 | Pruning Pattern의 시각화 | 희소도, 프루닝 |
| 13 | 026-027 | 가중치 분포 시각화 | 프루닝 |
| 14 | 028-029 | [실습 1] Fine-grained Pruning 구현 | 희소도, 프루닝, 마스크, fine-grained |
| 15 | 030-031 | Vector-level Pruning | 희소도, 프루닝, 마스크, fine-grained |
| 16 | 032-033 | Kernel-level Pruning | 희소도, 프루닝, 마스크, kernel-level |
| 17 | 034-035 | Channel-level Pruning | 희소도, 프루닝, 마스크, kernel-level |
| 18 | 036-037 | 중요도 합 비교 | 프루닝, 마스크, fine-grained, vector-level |
| 19 | 038-039 | Reconstruction Error 비교 | 프루닝, 마스크, fine-grained, vector-level |
| 20 | 040-041 | 1.2. Pruning Ratio | 희소도, 프루닝, 민감도 분석 |
| 21 | 042-045 | Sensitivity 분석 | DataLoader, 평가 루프, 모델 크기, 희소도 |
| 22 | 046-052 | [실습 2] Sensitivity Analysis를 통한 Pruning 수행 | DataLoader, 평가 루프, 모델 크기, 희소도 |
| 23 | 053-059 | [실습 3] Global Magnitude Pruning 구현 | DataLoader, 평가 루프, 모델 크기, 희소도 |
| 24 | 060-061 | 1.3. Pruning Schedule | 희소도, 프루닝, fine-grained, 스케줄링 |
| 25 | 062-063 | One-shot Pruning | DataLoader, 평가 루프, 학습 루프, 모델 크기 |
| 26 | 064-066 | [실습 4] Sparsity Scheduler 구현 | 희소도, 프루닝, fine-grained, 스케줄링 |
| 27 | 067-071 | Iterative Pruning | DataLoader, 평가 루프, 학습 루프, 모델 크기 |

## 2. 셀별 Walkthrough

아래 번호는 오른쪽 노트북의 cell 순서와 맞춘 것이다. Markdown 셀도 건너뛰지 않는다. Markdown 셀은 바로 다음 코드가 어떤 문제를 푸는지 정의하는 경우가 많기 때문이다.

### Cell 001 · Markdown · Assignment 1. Pruning for CNN

- **현재 구간**: Assignment 1. Pruning for CNN
- **오른쪽에서 읽을 내용**: # Assignment 1. Pruning for CNN
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 002 · Markdown · Goals

- **현재 구간**: Goals
- **오른쪽에서 읽을 내용**: ## Goals 본 실습은 CNN 모델의 다양한 Pruning 기법을 이해하고, 이를 구현하여 각 방법이 모델의 정확도와 효율성에 어떤 영향을 주는지 확인하는 것을 목표로 합니다. ## Contents 1. **Pruning Granularity:** - Fine-grained, Vector-level, Kernel-level, Channel-level의 차이를 학습하고 구현합니다. 2. **Pruning Ratio:** - 제거할 가중치의 비율(Sparsity)을 결정하는 방법을 배웁니다. Layer-wise와 Global 방식의 차이를 실습을 통해 확인합니다. 3. **Pruning Schedule:** - One-shot Pruni…
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - fine-grained는 개별 weight 단위다. 정확도는 잘 보존될 수 있지만 index가 불규칙해서 일반 dense kernel에서는 바로 빨라지지 않는다.
  - structured pruning 단위가 커질수록 실제 tensor 차원을 줄이거나 0-vector를 건너뛰기 쉬워진다. 대신 한 번에 제거되는 정보량이 커져 정확도 손실 위험이 커진다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 003 · Markdown · Setup

- **현재 구간**: Setup
- **오른쪽에서 읽을 내용**: # Setup
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - 주변 셀과 이어지는 보조 단계다. 새로 생기는 변수 이름과 다음 셀에서 소비되는 값을 연결해서 보면 흐름이 끊기지 않는다.
- **수학/shape 관점**:
  - $W_{pruned}=W\odot M,\; M\in\{0,1\}^{shape(W)}$
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 004 · Markdown · 다음 단계에 따라 실습 환경을 준비합니다. - **필수 모듈 가져오기**: 실습을 위한 필요한 라이브러리들을 import 합니다. - **Seed 설정**: 결과 재현성을 위해 난수 생성 Seed를 설정합니다. - **CIFAR-10 데이터셋 준비**: CIFAR-10 데이터셋을 다운로드하고 DataLoader를 생성합니다. - **사전 학습된 모델 로딩**: 미리 학습된 CNN 모델(VGG…

- **현재 구간**: Setup
- **오른쪽에서 읽을 내용**: 다음 단계에 따라 실습 환경을 준비합니다. - **필수 모듈 가져오기**: 실습을 위한 필요한 라이브러리들을 import 합니다. - **Seed 설정**: 결과 재현성을 위해 난수 생성 Seed를 설정합니다. - **CIFAR-10 데이터셋 준비**: CIFAR-10 데이터셋을 다운로드하고 DataLoader를 생성합니다. - **사전 학습된 모델 로딩**: 미리 학습된 CNN 모델(VGG 기반)을 로드합니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - 난수 seed를 고정한다. pruning threshold, data augmentation, optimizer 순서가 바뀌면 비교가 흔들리므로 baseline과 실험 조건을 고정하는 역할이다.
  - 데이터셋을 mini-batch 단위로 모델에 공급한다. CNN은 이미지 batch `[B,3,32,32]`, LLM은 token sequence `[B,T]`로 들어간다는 점을 구분한다.
- **수학/shape 관점**:
  - CNN 쪽 tensor는 이미지 `[B,3,32,32]`, feature map `[B,C,H,W]`, conv weight `[C_out,C_in,K_h,K_w]`로 읽는다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 005 · Code · import math import random from typing import List import numpy as np import torch from datasets import load_dataset from matplotlib import pyplot as plt from torch import nn from torch.optim import Optimizer from torch.o…

- **현재 구간**: Setup
- **오른쪽에서 볼 코드**: `import math import random from typing import List import numpy as np import torch from datasets import load_dataset from matplotlib import pyplot as plt from torch import nn from torch.optim import Optimizer from torch.o…`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 데이터셋을 mini-batch 단위로 모델에 공급한다. CNN은 이미지 batch `[B,3,32,32]`, LLM은 token sequence `[B,T]`로 들어간다는 점을 구분한다.
  - 학습/미세조정 루프다. forward → loss → backward → optimizer step 순서이며, pruning이면 step 이후 mask 유지, KD면 여러 loss 항의 비율을 확인한다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.
- **키워드**: DataLoader, 스케줄링, OPT

### Cell 006 · Code · Seed 설정

- **현재 구간**: Setup
- **오른쪽에서 볼 코드**: `# Seed 설정 random.seed(0) np.random.seed(0) torch.manual_seed(0)`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 난수 seed를 고정한다. pruning threshold, data augmentation, optimizer 순서가 바뀌면 비교가 흔들리므로 baseline과 실험 조건을 고정하는 역할이다.
- **수학/shape 관점**:
  - $W_{pruned}=W\odot M,\; M\in\{0,1\}^{shape(W)}$
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.
- **키워드**: 재현성

### Cell 007 · Markdown · CIFAR-10 데이터셋 준비

- **현재 구간**: CIFAR-10 데이터셋 준비
- **오른쪽에서 읽을 내용**: ## CIFAR-10 데이터셋 준비 CIFAR-10 데이터셋을 다운로드하고 DataLoader를 설정합니다. CIFAR-10은 32x32 크기의 RGB 이미지로 구성된 데이터셋으로, 비행기, 자동차, 새 등 10개의 클래스에 대해 각각 6,000개씩 총 60,000개의 이미지를 포함하고 있습니다. 이 중 50,000개는 학습용(train), 10,000개는 테스트용(test)으로 분리되어 있습니다. 데이터 증강(Data Augmentation)을 위해 학습 데이터에는 RandomCrop과 RandomHorizontalFlip을 적용하고, DataLoader를 통해 배치 단위로 효율적인 학습이 가능하도록 구성합니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - 데이터셋을 mini-batch 단위로 모델에 공급한다. CNN은 이미지 batch `[B,3,32,32]`, LLM은 token sequence `[B,T]`로 들어간다는 점을 구분한다.
  - 학습/미세조정 루프다. forward → loss → backward → optimizer step 순서이며, pruning이면 step 이후 mask 유지, KD면 여러 loss 항의 비율을 확인한다.
- **수학/shape 관점**:
  - CNN 쪽 tensor는 이미지 `[B,3,32,32]`, feature map `[B,C,H,W]`, conv weight `[C_out,C_in,K_h,K_w]`로 읽는다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 008 · Code · torchvision의 공식 다운로드 서버 에러를 피하기 위해

- **현재 구간**: CIFAR-10 데이터셋 준비
- **오른쪽에서 볼 코드**: `image_size = 32 / transforms = { / class HFCIFAR10(Dataset):`
- **정의되는 class**: `HFCIFAR10`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 데이터셋을 mini-batch 단위로 모델에 공급한다. CNN은 이미지 batch `[B,3,32,32]`, LLM은 token sequence `[B,T]`로 들어간다는 점을 구분한다.
  - 학습/미세조정 루프다. forward → loss → backward → optimizer step 순서이며, pruning이면 step 이후 mask 유지, KD면 여러 loss 항의 비율을 확인한다.
- **수학/shape 관점**:
  - CNN 쪽 tensor는 이미지 `[B,3,32,32]`, feature map `[B,C,H,W]`, conv weight `[C_out,C_in,K_h,K_w]`로 읽는다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.
- **키워드**: CIFAR-10 데이터, DataLoader

### Cell 009 · Markdown · 모델 로딩

- **현재 구간**: 모델 로딩
- **오른쪽에서 읽을 내용**: ## 모델 로딩 사전 학습된 CNN 모델을 불러옵니다. 이 모델은 Pruning 기법을 적용하기 전 기준 모델로 사용됩니다. 본 실습에서는 VGG 구조의 모델을 사용하며, CIFAR-10 데이터셋으로 미리 학습된 가중치를 활용합니다. 모델의 연산 효율을 높이기 위해 GPU에서 실행되도록 설정합니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - 데이터셋을 mini-batch 단위로 모델에 공급한다. CNN은 이미지 batch `[B,3,32,32]`, LLM은 token sequence `[B,T]`로 들어간다는 점을 구분한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - CNN 쪽 tensor는 이미지 `[B,3,32,32]`, feature map `[B,C,H,W]`, conv weight `[C_out,C_in,K_h,K_w]`로 읽는다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 010 · Code · TORCH_HUB_REPO = "SKKU-ESLAB/pytorch-models" / MODEL_NAME = "cifar10_vgg9_bn" / model = torch.hub.load(TORCH_HUB_REPO, MODEL_NAME, pretrained=True)

- **현재 구간**: 모델 로딩
- **오른쪽에서 볼 코드**: `TORCH_HUB_REPO = "SKKU-ESLAB/pytorch-models" / MODEL_NAME = "cifar10_vgg9_bn" / model = torch.hub.load(TORCH_HUB_REPO, MODEL_NAME, pretrained=True)`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 데이터셋을 mini-batch 단위로 모델에 공급한다. CNN은 이미지 batch `[B,3,32,32]`, LLM은 token sequence `[B,T]`로 들어간다는 점을 구분한다.
  - 학습/미세조정 루프다. forward → loss → backward → optimizer step 순서이며, pruning이면 step 이후 mask 유지, KD면 여러 loss 항의 비율을 확인한다.
- **수학/shape 관점**:
  - CNN 쪽 tensor는 이미지 `[B,3,32,32]`, feature map `[B,C,H,W]`, conv weight `[C_out,C_in,K_h,K_w]`로 읽는다.
- **직접 구현할 때 체크**:
  - 입력 tensor와 model parameter가 같은 device에 있는지 확인한다. CPU/GPU mismatch는 이 실습에서 흔한 오류다.
- **키워드**: CIFAR-10 데이터, VGG 모델

### Cell 011 · Markdown · 모델 크기 평가

- **현재 구간**: 모델 크기 평가
- **오른쪽에서 읽을 내용**: ## 모델 크기 평가 모델의 총 파라미터 개수를 계산하여 모델의 크기를 평가합니다. 각 파라미터의 element 수를 더하여 총 개수를 구하고, 이를 통해 모델의 크기를 비트 단위로 변환합니다. 데이터 너비는 일반적으로 32비트(float)를 사용합니다. 이를 통해 모델이 차지하는 메모리 용량을 확인하고 프루닝의 필요성을 강조할 수 있으며, 프루닝 후 크기가 얼마나 감소했는지 정량적으로 평가할 수 있습니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - 주변 셀과 이어지는 보조 단계다. 새로 생기는 변수 이름과 다음 셀에서 소비되는 값을 연결해서 보면 흐름이 끊기지 않는다.
- **수학/shape 관점**:
  - $W_{pruned}=W\odot M,\; M\in\{0,1\}^{shape(W)}$
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 012 · Code · def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int: / num_counted_elements = 0 / for param in model.parameters():

- **현재 구간**: 모델 크기 평가
- **오른쪽에서 볼 코드**: `def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int: / num_counted_elements = 0 / for param in model.parameters():`
- **정의되는 함수**: `get_num_parameters, get_model_size`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - quantization은 실수 범위를 정수 grid로 근사한다. `scale`은 grid 간격, `zero_point`는 실수 0이 대응되는 정수 위치다.
- **수학/shape 관점**:
  - quantization에서는 실수 tensor 자체보다 `(integer tensor, scale, zero_point/group_size)` 묶음이 실제 표현이다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.
- **키워드**: 모델 크기, zero-point

### Cell 013 · Markdown · 모델의 정확도 평가

- **현재 구간**: 모델의 정확도 평가
- **오른쪽에서 읽을 내용**: ## 모델의 정확도 평가 모델의 성능을 평가하기 위해 정확도를 측정합니다. 테스트 데이터셋을 사용하여 모델의 예측 정확성을 검증합니다. 모델 학습을 위한 `train` 함수와 성능 평가를 위한 `evaluate` 함수를 구현합니다. 이를 통해 프루닝 전후의 모델 정확도를 비교하여 성능 변화를 분석할 수 있습니다. 특히 프루닝으로 인한 정확도 손실과 모델 크기 감소의 trade-off 관계를 파악할 수 있습니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - 모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.
  - 학습/미세조정 루프다. forward → loss → backward → optimizer step 순서이며, pruning이면 step 이후 mask 유지, KD면 여러 loss 항의 비율을 확인한다.
- **수학/shape 관점**:
  - $W_{pruned}=W\odot M,\; M\in\{0,1\}^{shape(W)}$
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 014 · Code · def train( / callbacks=None, / model.train()

- **현재 구간**: 모델의 정확도 평가
- **오른쪽에서 볼 코드**: `def train( / callbacks=None, / model.train()`
- **정의되는 함수**: `train, evaluate`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 데이터셋을 mini-batch 단위로 모델에 공급한다. CNN은 이미지 batch `[B,3,32,32]`, LLM은 token sequence `[B,T]`로 들어간다는 점을 구분한다.
  - 모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.
  - 학습/미세조정 루프다. forward → loss → backward → optimizer step 순서이며, pruning이면 step 이후 mask 유지, KD면 여러 loss 항의 비율을 확인한다.
  - quantization은 실수 범위를 정수 grid로 근사한다. `scale`은 grid 간격, `zero_point`는 실수 0이 대응되는 정수 위치다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
  - quantization에서는 실수 tensor 자체보다 `(integer tensor, scale, zero_point/group_size)` 묶음이 실제 표현이다.
- **직접 구현할 때 체크**:
  - 입력 tensor와 model parameter가 같은 device에 있는지 확인한다. CPU/GPU mismatch는 이 실습에서 흔한 오류다.
  - 평가/압축 적용 단계에서는 gradient를 끄는 이유를 확인한다. 메모리 절약과 accidental gradient 방지를 위한 장치다.
- **키워드**: DataLoader, 평가 루프, 학습 루프, 스케줄링, zero-point, OPT

### Cell 015 · Code · dense_model_accuracy = evaluate(model, dataloader['test']) / print(f"dense model has accuracy={dense_model_accuracy:.2f}%")

- **현재 구간**: 모델의 정확도 평가
- **오른쪽에서 볼 코드**: `dense_model_accuracy = evaluate(model, dataloader['test']) / print(f"dense model has accuracy={dense_model_accuracy:.2f}%")`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 데이터셋을 mini-batch 단위로 모델에 공급한다. CNN은 이미지 batch `[B,3,32,32]`, LLM은 token sequence `[B,T]`로 들어간다는 점을 구분한다.
  - 모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.
- **수학/shape 관점**:
  - $W_{pruned}=W\odot M,\; M\in\{0,1\}^{shape(W)}$
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.
- **키워드**: DataLoader, 평가 루프

### Cell 016 · Markdown · 모델 가중치 값의 분포 확인

- **현재 구간**: 모델 가중치 값의 분포 확인
- **오른쪽에서 읽을 내용**: ## 모델 가중치 값의 분포 확인 모델의 가중치 분포를 시각적으로 확인하겠습니다. 가중치 값들을 히스토그램으로 표현하여 분포를 파악할 수 있습니다. 이는 프루닝(Pruning) 대상을 선정하는데 중요한 기준이 됩니다. 예를 들어, 매우 작은 값을 가진 가중치들은 프루닝 과정에서 우선적으로 제거될 수 있습니다. 여기서 `param.dim() > 1` 조건은 2차원 이상의 파라미터만 검사한다는 의미입니다. CNN에서 파라미터는 다음과 같이 구성됩니다: | Layer | Parameters | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; …
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - structured pruning 단위가 커질수록 실제 tensor 차원을 줄이거나 0-vector를 건너뛰기 쉬워진다. 대신 한 번에 제거되는 정보량이 커져 정확도 손실 위험이 커진다.
  - feature-level KD는 logits뿐 아니라 중간 representation을 맞춘다. teacher/student 채널 수가 다르면 regressor로 shape를 맞춘 뒤 MSE/Cosine 손실을 적용한다.
- **수학/shape 관점**:
  - CNN 쪽 tensor는 이미지 `[B,3,32,32]`, feature map `[B,C,H,W]`, conv weight `[C_out,C_in,K_h,K_w]`로 읽는다.
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 017 · Code · def plot_weight_distribution(model, bins=256, count_nonzero_only=False): / axes = axes.ravel() / plot_index = 0

- **현재 구간**: 모델 가중치 값의 분포 확인
- **오른쪽에서 볼 코드**: `def plot_weight_distribution(model, bins=256, count_nonzero_only=False): / axes = axes.ravel() / plot_index = 0`
- **정의되는 함수**: `plot_weight_distribution`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - quantization은 실수 범위를 정수 grid로 근사한다. `scale`은 grid 간격, `zero_point`는 실수 0이 대응되는 정수 위치다.
- **수학/shape 관점**:
  - CNN 쪽 tensor는 이미지 `[B,3,32,32]`, feature map `[B,C,H,W]`, conv weight `[C_out,C_in,K_h,K_w]`로 읽는다.
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
  - quantization에서는 실수 tensor 자체보다 `(integer tensor, scale, zero_point/group_size)` 묶음이 실제 표현이다.
- **직접 구현할 때 체크**:
  - reshape/transpose 후 어떤 축이 channel/group/kernel 축인지 반드시 주석으로 적어보는 것이 좋다.
- **키워드**: 모델 크기, zero-point, Linear layer

### Cell 018 · Markdown · 이로써 실습을 위한 기본적인 준비는 모두 마무리 되었습니다.

- **현재 구간**: 모델 가중치 값의 분포 확인
- **오른쪽에서 읽을 내용**: 이로써 실습을 위한 기본적인 준비는 모두 마무리 되었습니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - 주변 셀과 이어지는 보조 단계다. 새로 생기는 변수 이름과 다음 셀에서 소비되는 값을 연결해서 보면 흐름이 끊기지 않는다.
- **수학/shape 관점**:
  - $W_{pruned}=W\odot M,\; M\in\{0,1\}^{shape(W)}$
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 019 · Markdown · Reconstruction Error 계산 함수 정의

- **현재 구간**: Reconstruction Error 계산 함수 정의
- **오른쪽에서 읽을 내용**: ## Reconstruction Error 계산 함수 정의 실습에 들어가기 앞서 **reconstruction error**를 계산하는 함수를 정의하겠습니다. Reconstruction error는 프루닝 전후 모델 출력의 차이를 측정하여 프루닝이 모델의 성능에 미치는 영향을 정량적으로 평가하는 지표입니다. Input feature map $X$에 대해 원본 weight $W$와 pruning된 weight $\widehat{W}$ 사이의 reconstruction error는 다음 수식으로 계산됩니다: ${\left\Vert WX - \widehat{W}X \right\Vert}^2_2$ 이 값이 작을수록 프루닝 후에도 원래 모델의 출…
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - feature-level KD는 logits뿐 아니라 중간 representation을 맞춘다. teacher/student 채널 수가 다르면 regressor로 shape를 맞춘 뒤 MSE/Cosine 손실을 적용한다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 020 · Code · def get_reconstuction_error(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:

- **현재 구간**: Reconstruction Error 계산 함수 정의
- **오른쪽에서 볼 코드**: `def get_reconstuction_error(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:`
- **정의되는 함수**: `get_reconstuction_error`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 주변 셀과 이어지는 보조 단계다. 새로 생기는 변수 이름과 다음 셀에서 소비되는 값을 연결해서 보면 흐름이 끊기지 않는다.
- **수학/shape 관점**:
  - $W_{pruned}=W\odot M,\; M\in\{0,1\}^{shape(W)}$
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 021 · Markdown · 1.1. Pruning Granularity/Pattern

- **현재 구간**: 1.1. Pruning Granularity/Pattern
- **오른쪽에서 읽을 내용**: # 1.1. Pruning Granularity/Pattern 이 섹션에서는 Pruning을 수행할 때 다양한 granularity(세분성)를 적용하는 방법을 배웁니다. 각 granularity는 모델의 정확도(accuracy)와 하드웨어 효율성(hardware efficiency)에 영향을 미치므로, 적절한 선택이 중요합니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - 모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 022 · Markdown · Pruning Granularity 소개

- **현재 구간**: Pruning Granularity 소개
- **오른쪽에서 읽을 내용**: ## Pruning Granularity 소개 Pruning granularity는 Pruning이 적용되는 범위를 정의합니다. 다음과 같은 다양한 단위로 Pruning을 적용할 수 있습니다: - **Fine-grained Pruning**: 가장 작은 단위인 개별 가중치를 대상으로 합니다. - **Vector-level Pruning**: 가중치 벡터 단위로 Pruning을 수행합니다. - **Kernel-level Pruning**: 컨볼루션 커널 전체를 대상으로 합니다. - **Channel-level Pruning**: 전체 채널을 Pruning 대상으로 합니다. 각 단위는 Pruning이 모델의 구조와 성능에 미치는 영향을 다르…
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - fine-grained는 개별 weight 단위다. 정확도는 잘 보존될 수 있지만 index가 불규칙해서 일반 dense kernel에서는 바로 빨라지지 않는다.
  - structured pruning 단위가 커질수록 실제 tensor 차원을 줄이거나 0-vector를 건너뛰기 쉬워진다. 대신 한 번에 제거되는 정보량이 커져 정확도 손실 위험이 커진다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 023 · Markdown · Pruning Pattern의 시각화

- **현재 구간**: Pruning Pattern의 시각화
- **오른쪽에서 읽을 내용**: ## Pruning Pattern의 시각화 다음 그림은 다양한 Pruning pattern을 보여줍니다. 이를 통해 각 Pruning 기법의 특징을 시각적으로 이해할 수 있습니다. 그림
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 024 · Markdown · 본 실습에서는 가중치의 크기(Magnitude)를 기준으로 하는 중요도 점수를 사용합니다. Sparsity를 0.5로 설정하여 `model.backbone.conv1.weight`의 50%에 해당하는 가중치를 제거(Pruning)합니다. 이는 가중치의 절반을 0으로 만들어 모델을 희소화하는 과정입니다.

- **현재 구간**: Pruning Pattern의 시각화
- **오른쪽에서 읽을 내용**: 본 실습에서는 가중치의 크기(Magnitude)를 기준으로 하는 중요도 점수를 사용합니다. Sparsity를 0.5로 설정하여 `model.backbone.conv1.weight`의 50%에 해당하는 가중치를 제거(Pruning)합니다. 이는 가중치의 절반을 0으로 만들어 모델을 희소화하는 과정입니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - CNN 쪽 tensor는 이미지 `[B,3,32,32]`, feature map `[B,C,H,W]`, conv weight `[C_out,C_in,K_h,K_w]`로 읽는다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 025 · Code · prune_sparsity = 0.5 / weight = model.backbone.conv1.weight / print(weight.shape)

- **현재 구간**: Pruning Pattern의 시각화
- **오른쪽에서 볼 코드**: `prune_sparsity = 0.5 / weight = model.backbone.conv1.weight / print(weight.shape)`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - CNN 쪽 tensor는 이미지 `[B,3,32,32]`, feature map `[B,C,H,W]`, conv weight `[C_out,C_in,K_h,K_w]`로 읽는다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.
- **키워드**: 희소도, 프루닝

### Cell 026 · Markdown · 가중치 분포 시각화

- **현재 구간**: 가중치 분포 시각화
- **오른쪽에서 읽을 내용**: ## 가중치 분포 시각화 Pruning을 수행하기 전에 가중치의 분포를 시각화하여 중요한 가중치들을 파악하고 Pruning이 미치는 영향을 분석합니다. 아래 함수를 통해 가중치의 분포를 시각적으로 확인할 수 있습니다:
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 027 · Code · 가중치의 절댓값 분포 시각화

- **현재 구간**: 가중치 분포 시각화
- **오른쪽에서 볼 코드**: `def draw_weight_distribution(weight_tensor, title="Weight Distribution"): / weights = weight_tensor.detach().cpu().numpy() / if len(weights.shape) == 4:`
- **정의되는 함수**: `draw_weight_distribution`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 주변 셀과 이어지는 보조 단계다. 새로 생기는 변수 이름과 다음 셀에서 소비되는 값을 연결해서 보면 흐름이 끊기지 않는다.
- **수학/shape 관점**:
  - $W_{pruned}=W\odot M,\; M\in\{0,1\}^{shape(W)}$
- **직접 구현할 때 체크**:
  - reshape/transpose 후 어떤 축이 channel/group/kernel 축인지 반드시 주석으로 적어보는 것이 좋다.

### Cell 028 · Markdown · [실습 1] Fine-grained Pruning 구현

- **현재 구간**: [실습 1] Fine-grained Pruning 구현
- **오른쪽에서 읽을 내용**: ## [실습 1] Fine-grained Pruning 구현 Fine-grained Pruning은 신경망의 가중치를 개별적으로 제거하는 기법입니다. 각 가중치의 크기(Magnitude)를 기준으로 중요도를 평가하고, 지정된 희소도(sparsity)에 따라 중요도가 낮은 가중치들을 0으로 만듭니다. 이를 통해 모델의 크기를 줄이고 연산량을 감소시킬 수 있습니다. 아래 빈칸을 채워 `prune_weight_fine_grained` 함수를 완성하세요.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - fine-grained는 개별 weight 단위다. 정확도는 잘 보존될 수 있지만 index가 불규칙해서 일반 dense kernel에서는 바로 빨라지지 않는다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 029 · Code · 마스크 생성 및 시각화

- **현재 구간**: [실습 1] Fine-grained Pruning 구현
- **오른쪽에서 볼 코드**: `def prune_weight_fine_grained(weight: torch.Tensor, sparsity: float) -> None: / sparsity = min(1.0, max(0.0, sparsity)) / if sparsity == 1.0:  # 모든 가중치를 제거`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - fine-grained는 개별 weight 단위다. 정확도는 잘 보존될 수 있지만 index가 불규칙해서 일반 dense kernel에서는 바로 빨라지지 않는다.
  - quantization은 실수 범위를 정수 grid로 근사한다. `scale`은 grid 간격, `zero_point`는 실수 0이 대응되는 정수 위치다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
  - quantization에서는 실수 tensor 자체보다 `(integer tensor, scale, zero_point/group_size)` 묶음이 실제 표현이다.
- **직접 구현할 때 체크**:
  - `YOUR CODE` 위치는 직접 구현 대상이다. 오른쪽 코드의 앞뒤 변수 이름과 반환 shape를 먼저 맞춘다.
  - 비교 실험 전 원본 model state를 복구하는지 확인한다. pruning/quantization은 in-place 변경이 많다.
  - round 후 clamp 순서가 중요하다. 정수 범위를 넘으면 overflow/잘못된 dequantization이 생긴다.
- **키워드**: 희소도, 프루닝, 마스크, fine-grained, zero-point

### Cell 030 · Markdown · Vector-level Pruning

- **현재 구간**: Vector-level Pruning
- **오른쪽에서 읽을 내용**: ## Vector-level Pruning Vector-level Pruning은 가중치 벡터 단위로 Pruning을 수행하는 기법입니다. 이 방법은 fine-grained Pruning과 비교했을 때 구조적인 정보를 더 잘 보존할 수 있다는 장점이 있습니다. 각 가중치 벡터의 L1 norm(절대값의 합)을 기준으로 중요도를 평가하고, 중요도가 낮은 벡터들을 제거함으로써 Pruning을 수행합니다. 이러한 방식으로 네트워크의 구조적 특성을 유지하면서 모델을 경량화할 수 있습니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - fine-grained는 개별 weight 단위다. 정확도는 잘 보존될 수 있지만 index가 불규칙해서 일반 dense kernel에서는 바로 빨라지지 않는다.
  - structured pruning 단위가 커질수록 실제 tensor 차원을 줄이거나 0-vector를 건너뛰기 쉬워진다. 대신 한 번에 제거되는 정보량이 커져 정확도 손실 위험이 커진다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 031 · Code · def prune_weight_vector_level(weight: torch.Tensor, sparsity: float) -> None: / sparsity = min(1.0, max(0.0, sparsity)) / if sparsity == 1.0:  # 모든 가중치를 제거

- **현재 구간**: Vector-level Pruning
- **오른쪽에서 볼 코드**: `def prune_weight_vector_level(weight: torch.Tensor, sparsity: float) -> None: / sparsity = min(1.0, max(0.0, sparsity)) / if sparsity == 1.0:  # 모든 가중치를 제거`
- **정의되는 함수**: `prune_weight_vector_level`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - structured pruning 단위가 커질수록 실제 tensor 차원을 줄이거나 0-vector를 건너뛰기 쉬워진다. 대신 한 번에 제거되는 정보량이 커져 정확도 손실 위험이 커진다.
  - quantization은 실수 범위를 정수 grid로 근사한다. `scale`은 grid 간격, `zero_point`는 실수 0이 대응되는 정수 위치다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
  - quantization에서는 실수 tensor 자체보다 `(integer tensor, scale, zero_point/group_size)` 묶음이 실제 표현이다.
- **직접 구현할 때 체크**:
  - 비교 실험 전 원본 model state를 복구하는지 확인한다. pruning/quantization은 in-place 변경이 많다.
  - round 후 clamp 순서가 중요하다. 정수 범위를 넘으면 overflow/잘못된 dequantization이 생긴다.
- **키워드**: 희소도, 프루닝, 마스크, vector-level, zero-point

### Cell 032 · Markdown · Kernel-level Pruning

- **현재 구간**: Kernel-level Pruning
- **오른쪽에서 읽을 내용**: ## Kernel-level Pruning Kernel-level Pruning은 컨볼루션 커널을 기준으로 프루닝을 수행하는 방법입니다. 벡터 단위 프루닝보다 더 큰 구조적 단위로 프루닝을 수행하기 때문에 하드웨어 가속과 연산 최적화에 더 효과적입니다. 각 커널의 가중치 절댓값 총합을 중요도로 사용하여 불필요한 커널을 제거합니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - structured pruning 단위가 커질수록 실제 tensor 차원을 줄이거나 0-vector를 건너뛰기 쉬워진다. 대신 한 번에 제거되는 정보량이 커져 정확도 손실 위험이 커진다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 033 · Code · def prune_weight_kernel_level(weight: torch.Tensor, sparsity: float) -> None: / sparsity = min(1.0, max(0.0, sparsity)) / if sparsity == 1.0:

- **현재 구간**: Kernel-level Pruning
- **오른쪽에서 볼 코드**: `def prune_weight_kernel_level(weight: torch.Tensor, sparsity: float) -> None: / sparsity = min(1.0, max(0.0, sparsity)) / if sparsity == 1.0:`
- **정의되는 함수**: `prune_weight_kernel_level`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - structured pruning 단위가 커질수록 실제 tensor 차원을 줄이거나 0-vector를 건너뛰기 쉬워진다. 대신 한 번에 제거되는 정보량이 커져 정확도 손실 위험이 커진다.
  - quantization은 실수 범위를 정수 grid로 근사한다. `scale`은 grid 간격, `zero_point`는 실수 0이 대응되는 정수 위치다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
  - quantization에서는 실수 tensor 자체보다 `(integer tensor, scale, zero_point/group_size)` 묶음이 실제 표현이다.
- **직접 구현할 때 체크**:
  - 비교 실험 전 원본 model state를 복구하는지 확인한다. pruning/quantization은 in-place 변경이 많다.
  - round 후 clamp 순서가 중요하다. 정수 범위를 넘으면 overflow/잘못된 dequantization이 생긴다.
- **키워드**: 희소도, 프루닝, 마스크, kernel-level, channel-level, zero-point

### Cell 034 · Markdown · Channel-level Pruning

- **현재 구간**: Channel-level Pruning
- **오른쪽에서 읽을 내용**: ## Channel-level Pruning Channel-level Pruning은 입력 또는 출력 채널 전체를 Pruning 대상으로 합니다. 채널 단위로 프루닝을 수행하면 해당 채널과 관련된 모든 연산이 제거되므로 모델의 효율성을 향상시킬 수 있습니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - structured pruning 단위가 커질수록 실제 tensor 차원을 줄이거나 0-vector를 건너뛰기 쉬워진다. 대신 한 번에 제거되는 정보량이 커져 정확도 손실 위험이 커진다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 035 · Code · def prune_weight_channel_level(weight: torch.Tensor, sparsity: float) -> None: / sparsity = min(1.0, max(0.0, sparsity)) / if sparsity == 1.0:

- **현재 구간**: Channel-level Pruning
- **오른쪽에서 볼 코드**: `def prune_weight_channel_level(weight: torch.Tensor, sparsity: float) -> None: / sparsity = min(1.0, max(0.0, sparsity)) / if sparsity == 1.0:`
- **정의되는 함수**: `prune_weight_channel_level`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - structured pruning 단위가 커질수록 실제 tensor 차원을 줄이거나 0-vector를 건너뛰기 쉬워진다. 대신 한 번에 제거되는 정보량이 커져 정확도 손실 위험이 커진다.
  - quantization은 실수 범위를 정수 grid로 근사한다. `scale`은 grid 간격, `zero_point`는 실수 0이 대응되는 정수 위치다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
  - quantization에서는 실수 tensor 자체보다 `(integer tensor, scale, zero_point/group_size)` 묶음이 실제 표현이다.
- **직접 구현할 때 체크**:
  - 비교 실험 전 원본 model state를 복구하는지 확인한다. pruning/quantization은 in-place 변경이 많다.
  - round 후 clamp 순서가 중요하다. 정수 범위를 넘으면 overflow/잘못된 dequantization이 생긴다.
- **키워드**: 희소도, 프루닝, 마스크, kernel-level, channel-level, zero-point

### Cell 036 · Markdown · 중요도 합 비교

- **현재 구간**: 중요도 합 비교
- **오른쪽에서 읽을 내용**: ## 중요도 합 비교 각 프루닝 기법에 따라 제거된 가중치의 중요도 합을 비교합니다. 이를 통해 각 프루닝 방법이 얼마나 중요한 가중치를 제거하는지 파악할 수 있습니다. 중요도가 높은 가중치를 많이 제거할수록 모델의 성능 저하가 클 것으로 예상됩니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - 주변 셀과 이어지는 보조 단계다. 새로 생기는 변수 이름과 다음 셀에서 소비되는 값을 연결해서 보면 흐름이 끊기지 않는다.
- **수학/shape 관점**:
  - $W_{pruned}=W\odot M,\; M\in\{0,1\}^{shape(W)}$
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 037 · Code · 가중치의 중요도를 절댓값으로 계산

- **현재 구간**: 중요도 합 비교
- **오른쪽에서 볼 코드**: `importance = weight.abs() / importance_sum = importance.sum().item()  # 원본 / importance_sum_fine_grained = (importance * mask_fine_grained).sum().item()  # 미세 단위`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - fine-grained는 개별 weight 단위다. 정확도는 잘 보존될 수 있지만 index가 불규칙해서 일반 dense kernel에서는 바로 빨라지지 않는다.
  - structured pruning 단위가 커질수록 실제 tensor 차원을 줄이거나 0-vector를 건너뛰기 쉬워진다. 대신 한 번에 제거되는 정보량이 커져 정확도 손실 위험이 커진다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.
- **키워드**: 프루닝, 마스크, fine-grained, vector-level, kernel-level, channel-level

### Cell 038 · Markdown · Reconstruction Error 비교

- **현재 구간**: Reconstruction Error 비교
- **오른쪽에서 읽을 내용**: ## Reconstruction Error 비교 각 Pruning 기법을 적용한 후, 모델 출력의 reconstruction error를 비교합니다. 이를 통해 각 Pruning 방법이 모델의 출력에 미치는 영향을 정량적으로 평가할 수 있습니다. Reconstruction error가 클수록 원본 모델의 출력과 더 많이 달라졌음을 의미하며, 이는 해당 Pruning 방법이 모델의 성능에 더 큰 영향을 미칠 수 있음을 나타냅니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 039 · Code · 랜덤 입력 샘플 생성

- **현재 구간**: Reconstruction Error 비교
- **오른쪽에서 볼 코드**: `input_sample = torch.randn(1, 64, 32, 32, device=weight.device) / conv_original = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False, device=weight.device) / conv_original.weight.data = weight.clone()`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - fine-grained는 개별 weight 단위다. 정확도는 잘 보존될 수 있지만 index가 불규칙해서 일반 dense kernel에서는 바로 빨라지지 않는다.
  - structured pruning 단위가 커질수록 실제 tensor 차원을 줄이거나 0-vector를 건너뛰기 쉬워진다. 대신 한 번에 제거되는 정보량이 커져 정확도 손실 위험이 커진다.
- **수학/shape 관점**:
  - CNN 쪽 tensor는 이미지 `[B,3,32,32]`, feature map `[B,C,H,W]`, conv weight `[C_out,C_in,K_h,K_w]`로 읽는다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 비교 실험 전 원본 model state를 복구하는지 확인한다. pruning/quantization은 in-place 변경이 많다.
- **키워드**: 프루닝, 마스크, fine-grained, vector-level, kernel-level, channel-level

### Cell 040 · Markdown · 1.2. Pruning Ratio

- **현재 구간**: 1.2. Pruning Ratio
- **오른쪽에서 읽을 내용**: # 1.2. Pruning Ratio
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 041 · Markdown · - Layer별 민감도(Sensitivity) 분석 - 각 레이어의 중요도를 파악하기 위해 특정 레이어만 sparsity를 변화시키면서 정확도 변화를 측정 - 이를 통해 pruning에 민감한 레이어와 덜 민감한 레이어를 파악 - Non-uniform vs Uniform Pruning 비교 - Uniform: 모든 레이어에 동일한 pruning ratio 적용 - Non-uniform: 레이…

- **현재 구간**: 1.2. Pruning Ratio
- **오른쪽에서 읽을 내용**: - Layer별 민감도(Sensitivity) 분석 - 각 레이어의 중요도를 파악하기 위해 특정 레이어만 sparsity를 변화시키면서 정확도 변화를 측정 - 이를 통해 pruning에 민감한 레이어와 덜 민감한 레이어를 파악 - Non-uniform vs Uniform Pruning 비교 - Uniform: 모든 레이어에 동일한 pruning ratio 적용 - Non-uniform: 레이어별 민감도에 따라 global threshold 기반으로 다른 pruning ratio 적용 - 두 방식 간의 정확도와 모델 크기의 trade-off 비교 분석
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - sensitivity scan은 “어느 layer를 얼마나 잘라도 되는가”를 실험으로 찾는 단계다. 같은 sparsity라도 layer별 민감도가 달라 uniform pruning보다 non-uniform 설계가 유리할 수 있다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 042 · Markdown · Sensitivity 분석

- **현재 구간**: Sensitivity 분석
- **오른쪽에서 읽을 내용**: ## Sensitivity 분석 각 레이어의 민감도를 분석하기 위해 다른 레이어는 고정한 상태에서 특정 레이어의 sparsity를 변화시키며 정확도 변화를 측정합니다. 이를 통해 각 레이어가 pruning에 얼마나 민감한지 파악할 수 있습니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - sensitivity scan은 “어느 layer를 얼마나 잘라도 되는가”를 실험으로 찾는 단계다. 같은 sparsity라도 layer별 민감도가 달라 uniform pruning보다 non-uniform 설계가 유리할 수 있다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 043 · Code · 민감도 분석 실행

- **현재 구간**: Sensitivity 분석
- **오른쪽에서 볼 코드**: `@torch.no_grad() / def sensitivity_scan(model, dataloader, scan_step=0.1, scan_start=0.4, scan_end=1.0, verbose=True): / sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)`
- **정의되는 함수**: `sensitivity_scan`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 데이터셋을 mini-batch 단위로 모델에 공급한다. CNN은 이미지 batch `[B,3,32,32]`, LLM은 token sequence `[B,T]`로 들어간다는 점을 구분한다.
  - 모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - fine-grained는 개별 weight 단위다. 정확도는 잘 보존될 수 있지만 index가 불규칙해서 일반 dense kernel에서는 바로 빨라지지 않는다.
- **수학/shape 관점**:
  - CNN 쪽 tensor는 이미지 `[B,3,32,32]`, feature map `[B,C,H,W]`, conv weight `[C_out,C_in,K_h,K_w]`로 읽는다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 비교 실험 전 원본 model state를 복구하는지 확인한다. pruning/quantization은 in-place 변경이 많다.
  - 평가/압축 적용 단계에서는 gradient를 끄는 이유를 확인한다. 메모리 절약과 accidental gradient 방지를 위한 장치다.
- **키워드**: DataLoader, 평가 루프, 모델 크기, 희소도, 프루닝, fine-grained

### Cell 044 · Markdown · 각 레이어별 sparsity 변화에 따른 정확도를 시각화하여 sensitivity를 분석합니다. 아래 그래프는 각 레이어의 sparsity 변화에 따른 정확도 변화를 보여주며, 이를 통해 각 레이어의 pruning 민감도를 파악할 수 있습니다.

- **현재 구간**: Sensitivity 분석
- **오른쪽에서 읽을 내용**: 각 레이어별 sparsity 변화에 따른 정확도를 시각화하여 sensitivity를 분석합니다. 아래 그래프는 각 레이어의 sparsity 변화에 따른 정확도 변화를 보여주며, 이를 통해 각 레이어의 pruning 민감도를 파악할 수 있습니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - sensitivity scan은 “어느 layer를 얼마나 잘라도 되는가”를 실험으로 찾는 단계다. 같은 sparsity라도 layer별 민감도가 달라 uniform pruning보다 non-uniform 설계가 유리할 수 있다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 045 · Code · def plot_sensitivity_scan(sparsities, accuracies, dense_model_accuracy): / lower_bound_accuracy = dense_model_accuracy * 0.96  # 기준 정확도의 96%를 하한선으로 설정 / axes = axes.ravel()

- **현재 구간**: Sensitivity 분석
- **오른쪽에서 볼 코드**: `def plot_sensitivity_scan(sparsities, accuracies, dense_model_accuracy): / lower_bound_accuracy = dense_model_accuracy * 0.96  # 기준 정확도의 96%를 하한선으로 설정 / axes = axes.ravel()`
- **정의되는 함수**: `plot_sensitivity_scan`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - sensitivity scan은 “어느 layer를 얼마나 잘라도 되는가”를 실험으로 찾는 단계다. 같은 sparsity라도 layer별 민감도가 달라 uniform pruning보다 non-uniform 설계가 유리할 수 있다.
- **수학/shape 관점**:
  - CNN 쪽 tensor는 이미지 `[B,3,32,32]`, feature map `[B,C,H,W]`, conv weight `[C_out,C_in,K_h,K_w]`로 읽는다.
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.
- **키워드**: 모델 크기, 희소도, 프루닝, 민감도 분석, Linear layer

### Cell 046 · Markdown · [실습 2] Sensitivity Analysis를 통한 Pruning 수행

- **현재 구간**: [실습 2] Sensitivity Analysis를 통한 Pruning 수행
- **오른쪽에서 읽을 내용**: ## [실습 2] Sensitivity Analysis를 통한 Pruning 수행 Sensitivity analysis 결과를 바탕으로 각 레이어별 sparsity를 `custom_sparsity_dict`에 입력하세요. 위 그래프의 분석 결과를 토대로 아래와 같이 진행하면 됩니다다: - 정확도 하락이 적은 레이어: 높은 sparsity 값 설정 - 정확도 하락이 큰 레이어: 낮은 sparsity 값 설정 이러한 레이어별 차등적인 pruning을 통해 모델의 전반적인 성능은 유지하면서 효율적인 압축을 달성하는 것이 목표입니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - sensitivity scan은 “어느 layer를 얼마나 잘라도 되는가”를 실험으로 찾는 단계다. 같은 sparsity라도 layer별 민감도가 달라 uniform pruning보다 non-uniform 설계가 유리할 수 있다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 047 · Code · class FineGrainedPruner: / def __init__(self, model, sparsity_dict): / self.masks = FineGrainedPruner.prune(model, sparsity_dict)

- **현재 구간**: [실습 2] Sensitivity Analysis를 통한 Pruning 수행
- **오른쪽에서 볼 코드**: `class FineGrainedPruner: / def __init__(self, model, sparsity_dict): / self.masks = FineGrainedPruner.prune(model, sparsity_dict)`
- **정의되는 class**: `FineGrainedPruner`
- **정의되는 함수**: `get_uniform_sparsity_dict`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - fine-grained는 개별 weight 단위다. 정확도는 잘 보존될 수 있지만 index가 불규칙해서 일반 dense kernel에서는 바로 빨라지지 않는다.
- **수학/shape 관점**:
  - CNN 쪽 tensor는 이미지 `[B,3,32,32]`, feature map `[B,C,H,W]`, conv weight `[C_out,C_in,K_h,K_w]`로 읽는다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 평가/압축 적용 단계에서는 gradient를 끄는 이유를 확인한다. 메모리 절약과 accidental gradient 방지를 위한 장치다.
- **키워드**: 모델 크기, 희소도, 프루닝, 마스크, fine-grained

### Cell 048 · Code · 프루닝 적용 및 평가

- **현재 구간**: [실습 2] Sensitivity Analysis를 통한 Pruning 수행
- **오른쪽에서 볼 코드**: `def get_model_sparsity(model: nn.Module) -> float: / for param in model.parameters(): / custom_sparsity_dict = {`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 데이터셋을 mini-batch 단위로 모델에 공급한다. CNN은 이미지 batch `[B,3,32,32]`, LLM은 token sequence `[B,T]`로 들어간다는 점을 구분한다.
  - 모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - fine-grained는 개별 weight 단위다. 정확도는 잘 보존될 수 있지만 index가 불규칙해서 일반 dense kernel에서는 바로 빨라지지 않는다.
- **수학/shape 관점**:
  - CNN 쪽 tensor는 이미지 `[B,3,32,32]`, feature map `[B,C,H,W]`, conv weight `[C_out,C_in,K_h,K_w]`로 읽는다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
  - quantization에서는 실수 tensor 자체보다 `(integer tensor, scale, zero_point/group_size)` 묶음이 실제 표현이다.
- **직접 구현할 때 체크**:
  - `YOUR CODE` 위치는 직접 구현 대상이다. 오른쪽 코드의 앞뒤 변수 이름과 반환 shape를 먼저 맞춘다.
  - 비교 실험 전 원본 model state를 복구하는지 확인한다. pruning/quantization은 in-place 변경이 많다.
- **키워드**: DataLoader, 평가 루프, 모델 크기, 희소도, 프루닝, fine-grained

### Cell 049 · Markdown · Sensitivity analysis를 통한 pruning 결과와 비교하기 위해 모든 레이어에 uniform sparsity를 적용했을 때의 accuracy를 측정합니다.

- **현재 구간**: [실습 2] Sensitivity Analysis를 통한 Pruning 수행
- **오른쪽에서 읽을 내용**: Sensitivity analysis를 통한 pruning 결과와 비교하기 위해 모든 레이어에 uniform sparsity를 적용했을 때의 accuracy를 측정합니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - 모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - sensitivity scan은 “어느 layer를 얼마나 잘라도 되는가”를 실험으로 찾는 단계다. 같은 sparsity라도 layer별 민감도가 달라 uniform pruning보다 non-uniform 설계가 유리할 수 있다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 050 · Code · 균일한 sparsity 값들을 생성 (0.4부터 0.9까지 0.1 간격)

- **현재 구간**: [실습 2] Sensitivity Analysis를 통한 Pruning 수행
- **오른쪽에서 볼 코드**: `uniform_sparsities = np.arange(start=0.4, stop=1.0, step=0.1) / accuracies = [] / for sparsity in uniform_sparsities:`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 데이터셋을 mini-batch 단위로 모델에 공급한다. CNN은 이미지 batch `[B,3,32,32]`, LLM은 token sequence `[B,T]`로 들어간다는 점을 구분한다.
  - 모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - fine-grained는 개별 weight 단위다. 정확도는 잘 보존될 수 있지만 index가 불규칙해서 일반 dense kernel에서는 바로 빨라지지 않는다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 비교 실험 전 원본 model state를 복구하는지 확인한다. pruning/quantization은 in-place 변경이 많다.
- **키워드**: DataLoader, 평가 루프, 희소도, 프루닝, fine-grained

### Cell 051 · Markdown · Layer-wise pruning(uniform sparsity)와 global pruning(sensitivity analysis) 방식의 정확도를 비교하는 그래프입니다.

- **현재 구간**: [실습 2] Sensitivity Analysis를 통한 Pruning 수행
- **오른쪽에서 읽을 내용**: Layer-wise pruning(uniform sparsity)와 global pruning(sensitivity analysis) 방식의 정확도를 비교하는 그래프입니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - sensitivity scan은 “어느 layer를 얼마나 잘라도 되는가”를 실험으로 찾는 단계다. 같은 sparsity라도 layer별 민감도가 달라 uniform pruning보다 non-uniform 설계가 유리할 수 있다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 052 · Code · plt.plot(uniform_sparsities, accuracies, 'o-', label="Layer-wise Pruning") / plt.plot([custom_sparsity], [custom_pruned_model_accuracy], 'ro', label='Global Pruning (w/ sensitivity)') / plt.xlabel('Sparsity')

- **현재 구간**: [실습 2] Sensitivity Analysis를 통한 Pruning 수행
- **오른쪽에서 볼 코드**: `plt.plot(uniform_sparsities, accuracies, 'o-', label="Layer-wise Pruning") / plt.plot([custom_sparsity], [custom_pruned_model_accuracy], 'ro', label='Global Pruning (w/ sensitivity)') / plt.xlabel('Sparsity')`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - sensitivity scan은 “어느 layer를 얼마나 잘라도 되는가”를 실험으로 찾는 단계다. 같은 sparsity라도 layer별 민감도가 달라 uniform pruning보다 non-uniform 설계가 유리할 수 있다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.
- **키워드**: 희소도, 프루닝, 민감도 분석

### Cell 053 · Markdown · [실습 3] Global Magnitude Pruning 구현

- **현재 구간**: [실습 3] Global Magnitude Pruning 구현
- **오른쪽에서 읽을 내용**: ## [실습 3] Global Magnitude Pruning 구현 이 실습에서는 Global Magnitude Pruning을 구현합니다. Global Magnitude Pruning은 모델의 모든 가중치를 하나의 집합으로 간주하고, 가중치의 절대값을 기준으로 중요도를 평가하여 전체 모델에서 가장 작은 가중치를 제거하는 방식입니다. 이는 각 레이어별로 균일한 비율로 프루닝하는 것이 아니라, 전체 모델에서 가중치의 크기에 따라 프루닝을 수행합니다. **구현 과정:** 1. 모든 레이어의 가중치를 하나의 벡터로 통합 2. 가중치의 절대값으로 중요도 계산 3. 프루닝 비율에 따른 임계값 설정 4. 임계값 미만의 가중치를 제거하는 마스크 생성…
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - fine-grained는 개별 weight 단위다. 정확도는 잘 보존될 수 있지만 index가 불규칙해서 일반 dense kernel에서는 바로 빨라지지 않는다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 054 · Code · class FineGrainedPrunerV2: / def __init__(self, model, sparsity, global_prune=False): / self.masks = FineGrainedPrunerV2.prune(model, sparsity, global_prune)

- **현재 구간**: [실습 3] Global Magnitude Pruning 구현
- **오른쪽에서 볼 코드**: `class FineGrainedPrunerV2: / def __init__(self, model, sparsity, global_prune=False): / self.masks = FineGrainedPrunerV2.prune(model, sparsity, global_prune)`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - fine-grained는 개별 weight 단위다. 정확도는 잘 보존될 수 있지만 index가 불규칙해서 일반 dense kernel에서는 바로 빨라지지 않는다.
  - quantization은 실수 범위를 정수 grid로 근사한다. `scale`은 grid 간격, `zero_point`는 실수 0이 대응되는 정수 위치다.
- **수학/shape 관점**:
  - CNN 쪽 tensor는 이미지 `[B,3,32,32]`, feature map `[B,C,H,W]`, conv weight `[C_out,C_in,K_h,K_w]`로 읽는다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
  - quantization에서는 실수 tensor 자체보다 `(integer tensor, scale, zero_point/group_size)` 묶음이 실제 표현이다.
- **직접 구현할 때 체크**:
  - `YOUR CODE` 위치는 직접 구현 대상이다. 오른쪽 코드의 앞뒤 변수 이름과 반환 shape를 먼저 맞춘다.
  - 평가/압축 적용 단계에서는 gradient를 끄는 이유를 확인한다. 메모리 절약과 accidental gradient 방지를 위한 장치다.
  - reshape/transpose 후 어떤 축이 channel/group/kernel 축인지 반드시 주석으로 적어보는 것이 좋다.
- **키워드**: 모델 크기, 희소도, 프루닝, 마스크, fine-grained, zero-point

### Cell 055 · Code · Golbal Magnitude Pruning 수행

- **현재 구간**: [실습 3] Global Magnitude Pruning 구현
- **오른쪽에서 볼 코드**: `pruner = FineGrainedPrunerV2(model, custom_sparsity, global_prune=True) / global_magnitude_pruned_model_accuracy = evaluate(model, dataloader['test']) / model.recover_model()`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 데이터셋을 mini-batch 단위로 모델에 공급한다. CNN은 이미지 batch `[B,3,32,32]`, LLM은 token sequence `[B,T]`로 들어간다는 점을 구분한다.
  - 모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - fine-grained는 개별 weight 단위다. 정확도는 잘 보존될 수 있지만 index가 불규칙해서 일반 dense kernel에서는 바로 빨라지지 않는다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 비교 실험 전 원본 model state를 복구하는지 확인한다. pruning/quantization은 in-place 변경이 많다.
- **키워드**: DataLoader, 평가 루프, 희소도, 프루닝, fine-grained

### Cell 056 · Code · plt.plot(uniform_sparsities, accuracies, 'o-', label="Layer-wise Pruning") / plt.plot([custom_sparsity], [custom_pruned_model_accuracy], 'ro', label='Global Pruning (w/ sensitivity)') / plt.plot([custom_sparsity], [globa

- **현재 구간**: [실습 3] Global Magnitude Pruning 구현
- **오른쪽에서 볼 코드**: `plt.plot(uniform_sparsities, accuracies, 'o-', label="Layer-wise Pruning") / plt.plot([custom_sparsity], [custom_pruned_model_accuracy], 'ro', label='Global Pruning (w/ sensitivity)') / plt.plot([custom_sparsity], [globa`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - sensitivity scan은 “어느 layer를 얼마나 잘라도 되는가”를 실험으로 찾는 단계다. 같은 sparsity라도 layer별 민감도가 달라 uniform pruning보다 non-uniform 설계가 유리할 수 있다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.
- **키워드**: 희소도, 프루닝, 민감도 분석

### Cell 057 · Markdown · Global magnitude pruning 방식으로 pruning한 모델의 정확도를 다른 pruning 방식과 비교 분석합니다. Layer-wise pruning, Global pruning (sensitivity 기반), Global magnitude pruning 세 방식의 성능을 비교하여 각 방식의 특징을 비교합니다.

- **현재 구간**: [실습 3] Global Magnitude Pruning 구현
- **오른쪽에서 읽을 내용**: Global magnitude pruning 방식으로 pruning한 모델의 정확도를 다른 pruning 방식과 비교 분석합니다. Layer-wise pruning, Global pruning (sensitivity 기반), Global magnitude pruning 세 방식의 성능을 비교하여 각 방식의 특징을 비교합니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - sensitivity scan은 “어느 layer를 얼마나 잘라도 되는가”를 실험으로 찾는 단계다. 같은 sparsity라도 layer별 민감도가 달라 uniform pruning보다 non-uniform 설계가 유리할 수 있다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 058 · Code · global_magnitude_pruned_model_accuracies = [] / for sparsity in uniform_sparsities: / pruner = FineGrainedPrunerV2(model, sparsity, global_prune=True)

- **현재 구간**: [실습 3] Global Magnitude Pruning 구현
- **오른쪽에서 볼 코드**: `global_magnitude_pruned_model_accuracies = [] / for sparsity in uniform_sparsities: / pruner = FineGrainedPrunerV2(model, sparsity, global_prune=True)`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 데이터셋을 mini-batch 단위로 모델에 공급한다. CNN은 이미지 batch `[B,3,32,32]`, LLM은 token sequence `[B,T]`로 들어간다는 점을 구분한다.
  - 모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - fine-grained는 개별 weight 단위다. 정확도는 잘 보존될 수 있지만 index가 불규칙해서 일반 dense kernel에서는 바로 빨라지지 않는다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 비교 실험 전 원본 model state를 복구하는지 확인한다. pruning/quantization은 in-place 변경이 많다.
- **키워드**: DataLoader, 평가 루프, 희소도, 프루닝, fine-grained

### Cell 059 · Code · plt.plot(uniform_sparsities, accuracies, 'o-', label="Layer-wise Pruning") / plt.plot([custom_sparsity], [custom_pruned_model_accuracy], 'ro', label='Global Pruning (w/ sensitivity)') / plt.plot(uniform_sparsities, globa

- **현재 구간**: [실습 3] Global Magnitude Pruning 구현
- **오른쪽에서 볼 코드**: `plt.plot(uniform_sparsities, accuracies, 'o-', label="Layer-wise Pruning") / plt.plot([custom_sparsity], [custom_pruned_model_accuracy], 'ro', label='Global Pruning (w/ sensitivity)') / plt.plot(uniform_sparsities, globa`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - sensitivity scan은 “어느 layer를 얼마나 잘라도 되는가”를 실험으로 찾는 단계다. 같은 sparsity라도 layer별 민감도가 달라 uniform pruning보다 non-uniform 설계가 유리할 수 있다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.
- **키워드**: 희소도, 프루닝, 민감도 분석

### Cell 060 · Markdown · 1.3. Pruning Schedule

- **현재 구간**: 1.3. Pruning Schedule
- **오른쪽에서 읽을 내용**: # 1.3. Pruning Schedule
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 061 · Markdown · 이 섹션에서는 Pruning 스케줄링을 구현하고, 모델의 Fine-tuning과 함께 Pruning을 진행합니다. Pruning 스케줄링은 모델 학습 중 특정 시점에 가중치를 제거하는 체계적인 접근 방식입니다. 이를 통해 모델의 학습 과정에서 점진적으로 Pruning을 적용하여 성능 저하를 최소화하면서 모델 크기를 효과적으로 줄일 수 있습니다. **주요 Pruning 스케줄 방식:** - *…

- **현재 구간**: 1.3. Pruning Schedule
- **오른쪽에서 읽을 내용**: 이 섹션에서는 Pruning 스케줄링을 구현하고, 모델의 Fine-tuning과 함께 Pruning을 진행합니다. Pruning 스케줄링은 모델 학습 중 특정 시점에 가중치를 제거하는 체계적인 접근 방식입니다. 이를 통해 모델의 학습 과정에서 점진적으로 Pruning을 적용하여 성능 저하를 최소화하면서 모델 크기를 효과적으로 줄일 수 있습니다. **주요 Pruning 스케줄 방식:** - **One-shot Pruning**: 목표 Pruning 비율을 한 번에 적용한 후, Fine-tuning으로 성능을 회복합니다. - **Iterative Pruning**: 여러 단계에 걸쳐 점진적으로 Pruning을 수행합니다. 각 단계마다 일정 …
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - 학습/미세조정 루프다. forward → loss → backward → optimizer step 순서이며, pruning이면 step 이후 mask 유지, KD면 여러 loss 항의 비율을 확인한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - fine-grained는 개별 weight 단위다. 정확도는 잘 보존될 수 있지만 index가 불규칙해서 일반 dense kernel에서는 바로 빨라지지 않는다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 062 · Markdown · One-shot Pruning

- **현재 구간**: One-shot Pruning
- **오른쪽에서 읽을 내용**: ## One-shot Pruning 실습 3에서 구현한 `FineGrainedPrunerV2`를 사용하여 One-shot Pruning을 구현합니다. 목표 sparsity를 95%로 설정하고 5 epoch 동안 fine-tuning을 진행합니다. 학습에 사용되는 주요 구성 요소: - Optimizer: SGD (learning rate=0.01, momentum=0.9, weight_decay=1e-4) - Scheduler: CosineAnnealingLR - Loss function: CrossEntropyLoss
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - 학습/미세조정 루프다. forward → loss → backward → optimizer step 순서이며, pruning이면 step 이후 mask 유지, KD면 여러 loss 항의 비율을 확인한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - fine-grained는 개별 weight 단위다. 정확도는 잘 보존될 수 있지만 index가 불규칙해서 일반 dense kernel에서는 바로 빨라지지 않는다.
  - feature-level KD는 logits뿐 아니라 중간 representation을 맞춘다. teacher/student 채널 수가 다르면 regressor로 shape를 맞춘 뒤 MSE/Cosine 손실을 적용한다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 063 · Code · num_finetune_epochs = 5 / target_sparsity = 0.95 / optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

- **현재 구간**: One-shot Pruning
- **오른쪽에서 볼 코드**: `num_finetune_epochs = 5 / target_sparsity = 0.95 / optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 데이터셋을 mini-batch 단위로 모델에 공급한다. CNN은 이미지 batch `[B,3,32,32]`, LLM은 token sequence `[B,T]`로 들어간다는 점을 구분한다.
  - 모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.
  - 학습/미세조정 루프다. forward → loss → backward → optimizer step 순서이며, pruning이면 step 이후 mask 유지, KD면 여러 loss 항의 비율을 확인한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 비교 실험 전 원본 model state를 복구하는지 확인한다. pruning/quantization은 in-place 변경이 많다.
- **키워드**: DataLoader, 평가 루프, 학습 루프, 모델 크기, 희소도, 프루닝

### Cell 064 · Markdown · [실습 4] Sparsity Scheduler 구현

- **현재 구간**: [실습 4] Sparsity Scheduler 구현
- **오른쪽에서 읽을 내용**: ## [실습 4] Sparsity Scheduler 구현 이 실습에서는 시간에 따른 Pruning 비율을 조절하는 다음 2가지 Sparsity Scheduler를 구현합니다: - **Linear sparsity scheduler**: $v^{(t)} = v_f + \left(v_i - v_f\right)\left(1 - \frac{t - t_i}{t_f - t_i}\right)$ - 시간에 따라 선형적으로 sparsity가 증가 - **Cubic sparsity scheduler**: $v^{(t)} = v_f + \left(v_i - v_f\right)\left(1 - \frac{t - t_i}{t_f - t_i}\right)^3$ -…
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - 학습/미세조정 루프다. forward → loss → backward → optimizer step 순서이며, pruning이면 step 이후 mask 유지, KD면 여러 loss 항의 비율을 확인한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 065 · Code · Linear sparsity schedule

- **현재 구간**: [실습 4] Sparsity Scheduler 구현
- **오른쪽에서 볼 코드**: `def get_sparsity_schedule(num_epochs: int, / sparsity_schedule = [] / for epoch in range(num_epochs):`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
- **직접 구현할 때 체크**:
  - `YOUR CODE` 위치는 직접 구현 대상이다. 오른쪽 코드의 앞뒤 변수 이름과 반환 shape를 먼저 맞춘다.
- **키워드**: 희소도, fine-grained, Linear layer

### Cell 066 · Code · plt.plot(range(num_finetune_epochs), linear_sparsity_schedule, 'o-', label='Linear') / plt.plot(range(num_finetune_epochs), cubic_sparsity_schedule, 'o-', label='Cubic') / plt.xticks(np.arange(5))

- **현재 구간**: [실습 4] Sparsity Scheduler 구현
- **오른쪽에서 볼 코드**: `plt.plot(range(num_finetune_epochs), linear_sparsity_schedule, 'o-', label='Linear') / plt.plot(range(num_finetune_epochs), cubic_sparsity_schedule, 'o-', label='Cubic') / plt.xticks(np.arange(5))`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.
- **키워드**: 희소도, fine-grained, Linear layer

### Cell 067 · Markdown · Iterative Pruning

- **현재 구간**: Iterative Pruning
- **오른쪽에서 읽을 내용**: ## Iterative Pruning Iterative Pruning은 모델의 가중치를 한 번에 제거하는 대신 여러 단계에 걸쳐 점진적으로 제거하는 방법입니다. 각 단계에서 일정 비율의 가중치를 제거한 후 모델을 재학습시켜 남은 가중치들이 제거된 가중치를 보완할 수 있도록 합니다. 이때 가중치를 제거하는 비율을 조절하는 방법으로 Linear와 Cubic Sparsity Scheduler를 사용할 수 있습니다. Cubic Scheduler는 초기에는 빠르게 가중치를 제거하다가 후반부로 갈수록 천천히 제거하는 방식을 사용합니다. 이는 모델이 후반에 중요한 가중치를 신중하게 제거할 수 있도록 시간을 더 많이 주기 때문에, Linear Sche…
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - 학습/미세조정 루프다. forward → loss → backward → optimizer step 순서이며, pruning이면 step 이후 mask 유지, KD면 여러 loss 항의 비율을 확인한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 068 · Markdown · Linear sparsity schedule을 사용하여 iterative pruning을 수행하세요.

- **현재 구간**: Iterative Pruning
- **오른쪽에서 읽을 내용**: Linear sparsity schedule을 사용하여 iterative pruning을 수행하세요.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 069 · Code · optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4) / scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_finetune_epochs * len(dataloader['train'])) / criterion =

- **현재 구간**: Iterative Pruning
- **오른쪽에서 볼 코드**: `optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4) / scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_finetune_epochs * len(dataloader['train'])) / criterion =`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 데이터셋을 mini-batch 단위로 모델에 공급한다. CNN은 이미지 batch `[B,3,32,32]`, LLM은 token sequence `[B,T]`로 들어간다는 점을 구분한다.
  - 모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.
  - 학습/미세조정 루프다. forward → loss → backward → optimizer step 순서이며, pruning이면 step 이후 mask 유지, KD면 여러 loss 항의 비율을 확인한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 비교 실험 전 원본 model state를 복구하는지 확인한다. pruning/quantization은 in-place 변경이 많다.
- **키워드**: DataLoader, 평가 루프, 학습 루프, 모델 크기, 희소도, 프루닝

### Cell 070 · Markdown · Cubic sparsity schedule을 사용하여 iterative pruning을 수행하세요.

- **현재 구간**: Iterative Pruning
- **오른쪽에서 읽을 내용**: Cubic sparsity schedule을 사용하여 iterative pruning을 수행하세요.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 071 · Code · optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4) / scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_finetune_epochs * len(dataloader['train'])) / criterion =

- **현재 구간**: Iterative Pruning
- **오른쪽에서 볼 코드**: `optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4) / scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_finetune_epochs * len(dataloader['train'])) / criterion =`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 데이터셋을 mini-batch 단위로 모델에 공급한다. CNN은 이미지 batch `[B,3,32,32]`, LLM은 token sequence `[B,T]`로 들어간다는 점을 구분한다.
  - 모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.
  - 학습/미세조정 루프다. forward → loss → backward → optimizer step 순서이며, pruning이면 step 이후 mask 유지, KD면 여러 loss 항의 비율을 확인한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 비교 실험 전 원본 model state를 복구하는지 확인한다. pruning/quantization은 in-place 변경이 많다.
- **키워드**: DataLoader, 평가 루프, 학습 루프, 모델 크기, 희소도, 프루닝

## 3. 실습 후 스스로 확인할 질문

1. 이 노트북에서 baseline metric은 무엇이고, 압축 후 얼마만큼 변했는가?
2. in-place로 model weight를 바꾸는 셀은 어디이며, 원본 복구/reset은 어떻게 하는가?
3. 핵심 함수 하나를 빈 파일에 다시 구현한다면 입력/출력 shape를 주석으로 쓸 수 있는가?
4. 정확도/PPL 손실이 생겼을 때 원인이 range 추정, mask 단위, scale 선택, calibration 부족 중 어디에 가까운가?
5. 실제 hardware speedup으로 이어지려면 단순 parameter 감소 외에 어떤 조건이 필요한가?
