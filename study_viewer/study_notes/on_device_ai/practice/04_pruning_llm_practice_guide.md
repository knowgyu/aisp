# On-Device AI Practice 04 — Pruning for LLM 셀별 코드 학습 가이드

> 오른쪽에는 원본 노트북 HTML을 띄우고, 왼쪽은 **모든 셀을 따라가는 해설 지도**로 쓴다. 이 문서는 원본 코드를 대체하지 않고, 각 셀이 왜 필요한지/무슨 shape로 움직이는지/직접 구현할 때 어디를 봐야 하는지 설명한다.

- 기준 교안: `ODAI-2 Chapter 1 LLM Pruning`
- 원본 노트북: `On-Device AI 강의자료/실습/4. Pruning for LLM.ipynb`
- 학습 목표: SmolLM 계열 causal LM에서 perplexity 평가, magnitude pruning, Wanda activation-aware pruning, N:M structured pruning을 연결한다.

## 0. 그림으로 먼저 잡기

```mermaid
flowchart LR
  A["Load LM/tokenizer"]
  B["Wikitext perplexity"]
  A --> B
  C["Magnitude pruning"]
  B --> C
  D["Calibration activations"]
  C --> D
  E["Wanda score"]
  D --> E
  F["N:M sparsity"]
  E --> F
```

### 실습 전체에서 계속 붙잡을 수식

$score_{ij}=|W_{ij}|\cdot \|X_j\|_2$  (Wanda의 핵심 직관)

### 핵심 shape 표

| 대상 | Shape / 표현 | 의미 |
|---|---|---|
| token ids | `[B, T]` | 언어 모델 입력 토큰 |
| hidden states | `[B, T, d_model]` | linear layer 입력 activation |
| linear weight | `[d_out, d_in]` | LLM pruning 주요 대상 |
| PPL | `exp(cross_entropy)` | pruning 후 품질 지표 |

## 1. 노트북 전체 지도

| 구간 | 셀 범위 | 오른쪽에서 보이는 제목 | 여기서 잡아야 할 것 |
|---:|---:|---|---|
| 1 | 001-001 | Assignment 4. Pruning for LLM | 프루닝 |
| 2 | 002-002 | Goals | 프루닝, Wanda, activation |
| 3 | 003-005 | Setup | tokenizer |
| 4 | 006-007 | 모델 평가 | 평가 루프, tokenizer, perplexity, Wikitext |
| 5 | 008-009 | SmolLM-135M 모델 로딩 | 평가 루프, tokenizer, perplexity |
| 6 | 010-013 | [실습 1] Magnitude-based Pruning 구현 | 평가 루프, 희소도, 프루닝, 마스크 |
| 7 | 014-018 | [실습 2] Calibration 데이터셋 준비 | 재현성, zero-point, calibration, tokenizer |
| 8 | 019-021 | [실습 3] Wanda Pruning 구현 | 평가 루프, 모델 크기, 희소도, 프루닝 |

## 2. 셀별 Walkthrough

아래 번호는 오른쪽 노트북의 cell 순서와 맞춘 것이다. Markdown 셀도 건너뛰지 않는다. Markdown 셀은 바로 다음 코드가 어떤 문제를 푸는지 정의하는 경우가 많기 때문이다.

### Cell 001 · Markdown · Assignment 4. Pruning for LLM

- **현재 구간**: Assignment 4. Pruning for LLM
- **오른쪽에서 읽을 내용**: # Assignment 4. Pruning for LLM
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 002 · Markdown · Goals

- **현재 구간**: Goals
- **오른쪽에서 읽을 내용**: ## Goals 본 실습에서는 대규모 언어 모델(Large Language Model, LLM)의 크기를 효과적으로 줄이는 Pruning 기법을 학습합니다. 특히 Magnitude-based pruning과 최근 주목받는 Wanda 기법을 활용하여, 파라미터의 수를 효율적으로 감소시키면서 모델의 성능을 유지하는 방법을 실습합니다. ## Contents 1. **Magnitude-based Pruning 실습**: - 간단한 magnitude 기반 pruning 방법을 통해 모델 크기를 감소시키고 성능 변화를 확인합니다. 2. **Wanda를 이용한 Pruning 실습**: - Activation의 중요도를 측정하여 더 정교하게 pruni…
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - Wanda는 weight magnitude만 보지 않고 입력 activation 크기도 곱한다. 같은 weight라도 자주/크게 활성화되는 channel에 연결되면 더 중요하다고 보는 방식이다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 003 · Markdown · Setup

- **현재 구간**: Setup
- **오른쪽에서 읽을 내용**: ## Setup
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - 주변 셀과 이어지는 보조 단계다. 새로 생기는 변수 이름과 다음 셀에서 소비되는 값을 연결해서 보면 흐름이 끊기지 않는다.
- **수학/shape 관점**:
  - $score_{ij}=|W_{ij}|\cdot \|X_j\|_2$  (Wanda의 핵심 직관)
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 004 · Markdown · 필요한 모듈을 불러옵니다.

- **현재 구간**: Setup
- **오른쪽에서 읽을 내용**: 필요한 모듈을 불러옵니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - 주변 셀과 이어지는 보조 단계다. 새로 생기는 변수 이름과 다음 셀에서 소비되는 값을 연결해서 보면 흐름이 끊기지 않는다.
- **수학/shape 관점**:
  - $score_{ij}=|W_{ij}|\cdot \|X_j\|_2$  (Wanda의 핵심 직관)
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 005 · Code · import tqdm import torch from torch import nn from transformers import AutoModelForCausalLM, AutoTokenizer from datasets import load_dataset from functools import partial import gc import os

- **현재 구간**: Setup
- **오른쪽에서 볼 코드**: `import tqdm import torch from torch import nn from transformers import AutoModelForCausalLM, AutoTokenizer from datasets import load_dataset from functools import partial import gc import os`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 데이터셋을 mini-batch 단위로 모델에 공급한다. CNN은 이미지 batch `[B,3,32,32]`, LLM은 token sequence `[B,T]`로 들어간다는 점을 구분한다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.
- **키워드**: tokenizer

### Cell 006 · Markdown · 모델 평가

- **현재 구간**: 모델 평가
- **오른쪽에서 읽을 내용**: ### 모델 평가 Wikitext-2 데이터셋을 사용하여 모델의 성능을 평가하는 지표인 perplexity를 계산합니다. **Perplexity란?** - Perplexity는 언어 모델이 주어진 텍스트를 얼마나 잘 예측하는지를 수치로 나타낸 지표입니다. - 수학적으로는 모델이 예측한 확률분포의 "불확실성"을 측정하는 값이며, 값이 **낮을수록 모델의 성능이 좋다**고 해석합니다. - 단어 $\{w_1, w_2, w_3, ..., w_N\}$으로 구성된 문장의 Perplexity는 다음과 같은 수식으로 나타낼 수 있습니다. - $Perplexity = \sqrt[N]{\frac{1}{\prod_{i=1}^{N} P(w_i | w_1, w…
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - 모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.
- **수학/shape 관점**:
  - PPL은 `exp(average cross entropy)`다. 낮을수록 다음 token 예측이 좋다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 007 · Code · def evaluate(model, tokenizer): / testenc = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test') / testenc = tokenizer("\n\n".join(testenc['text']), return_tensors='pt')

- **현재 구간**: 모델 평가
- **오른쪽에서 볼 코드**: `def evaluate(model, tokenizer): / testenc = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test') / testenc = tokenizer("\n\n".join(testenc['text']), return_tensors='pt')`
- **정의되는 함수**: `evaluate`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 데이터셋을 mini-batch 단위로 모델에 공급한다. CNN은 이미지 batch `[B,3,32,32]`, LLM은 token sequence `[B,T]`로 들어간다는 점을 구분한다.
  - 모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.
- **수학/shape 관점**:
  - PPL은 `exp(average cross entropy)`다. 낮을수록 다음 token 예측이 좋다.
- **직접 구현할 때 체크**:
  - 입력 tensor와 model parameter가 같은 device에 있는지 확인한다. CPU/GPU mismatch는 이 실습에서 흔한 오류다.
  - 평가/압축 적용 단계에서는 gradient를 끄는 이유를 확인한다. 메모리 절약과 accidental gradient 방지를 위한 장치다.
  - reshape/transpose 후 어떤 축이 channel/group/kernel 축인지 반드시 주석으로 적어보는 것이 좋다.
- **키워드**: 평가 루프, tokenizer, perplexity, Wikitext

### Cell 008 · Markdown · SmolLM-135M 모델 로딩

- **현재 구간**: SmolLM-135M 모델 로딩
- **오른쪽에서 읽을 내용**: ### SmolLM-135M 모델 로딩 SmolLM-135M 모델을 로딩하고 평가합니다. 여기서 tokenizer는 텍스트(문장)를 모델이 이해할 수 있는 작은 단위(token)로 나누는 역할을 수행합니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - 핵심 키워드: tokenizer. 이 셀에서 해당 키워드가 코드 변수/함수로 어떻게 구현되는지 오른쪽 노트북에서 확인한다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 009 · Code · Evaluate the model

- **현재 구간**: SmolLM-135M 모델 로딩
- **오른쪽에서 볼 코드**: `model_path = "HuggingFaceTB/SmolLM-135M" / tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False) / model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, us`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.
  - 학습/미세조정 루프다. forward → loss → backward → optimizer step 순서이며, pruning이면 step 이후 mask 유지, KD면 여러 loss 항의 비율을 확인한다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
  - PPL은 `exp(average cross entropy)`다. 낮을수록 다음 token 예측이 좋다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.
- **키워드**: 평가 루프, tokenizer, perplexity

### Cell 010 · Markdown · [실습 1] Magnitude-based Pruning 구현

- **현재 구간**: [실습 1] Magnitude-based Pruning 구현
- **오른쪽에서 읽을 내용**: ## [실습 1] Magnitude-based Pruning 구현 Magnitude-based pruning 함수를 구현하세요. 구현은 CNN에서 magnitude-based pruning을 구현하는 방식과 유사합니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 011 · Code · @torch.no_grad() / def prune_magnitude(model, sparsity): / for n, m in model.named_modules():

- **현재 구간**: [실습 1] Magnitude-based Pruning 구현
- **오른쪽에서 볼 코드**: `@torch.no_grad() / def prune_magnitude(model, sparsity): / for n, m in model.named_modules():`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - quantization은 실수 범위를 정수 grid로 근사한다. `scale`은 grid 간격, `zero_point`는 실수 0이 대응되는 정수 위치다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
  - quantization에서는 실수 tensor 자체보다 `(integer tensor, scale, zero_point/group_size)` 묶음이 실제 표현이다.
- **직접 구현할 때 체크**:
  - `YOUR CODE` 위치는 직접 구현 대상이다. 오른쪽 코드의 앞뒤 변수 이름과 반환 shape를 먼저 맞춘다.
  - 평가/압축 적용 단계에서는 gradient를 끄는 이유를 확인한다. 메모리 절약과 accidental gradient 방지를 위한 장치다.
- **키워드**: 희소도, 프루닝, 마스크, zero-point, Linear layer

### Cell 012 · Markdown · Pruning된 모델을 평가합니다.

- **현재 구간**: [실습 1] Magnitude-based Pruning 구현
- **오른쪽에서 읽을 내용**: Pruning된 모델을 평가합니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 013 · Code · Evaluate the model

- **현재 구간**: [실습 1] Magnitude-based Pruning 구현
- **오른쪽에서 볼 코드**: `model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, use_safetensors=True) / model_perplexity = evaluate(model, tokenizer) / print(f"\nmodel perplexity (magnitude 50%): {`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.
  - 학습/미세조정 루프다. forward → loss → backward → optimizer step 순서이며, pruning이면 step 이후 mask 유지, KD면 여러 loss 항의 비율을 확인한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
  - PPL은 `exp(average cross entropy)`다. 낮을수록 다음 token 예측이 좋다.
- **직접 구현할 때 체크**:
  - 입력 tensor와 model parameter가 같은 device에 있는지 확인한다. CPU/GPU mismatch는 이 실습에서 흔한 오류다.
- **키워드**: 평가 루프, 프루닝, tokenizer, perplexity

### Cell 014 · Markdown · [실습 2] Calibration 데이터셋 준비

- **현재 구간**: [실습 2] Calibration 데이터셋 준비
- **오른쪽에서 읽을 내용**: ## [실습 2] Calibration 데이터셋 준비 [Wanda](https://arxiv.org/pdf/2306.11695)를 사용하여 importance를 계산하기 위해서는 calibration 데이터셋을 통해 activation을 추출해야 합니다. 이를 위해 ```get_calib_dataset``` 함수는 calibration dataset을 준비합니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - calibration/observer는 activation range를 샘플 데이터로 추정한다. range가 좁으면 clipping, 넓으면 resolution 손실이 생긴다.
  - Wanda는 weight magnitude만 보지 않고 입력 activation 크기도 곱한다. 같은 weight라도 자주/크게 활성화되는 channel에 연결되면 더 중요하다고 보는 방식이다.
- **수학/shape 관점**:
  - $score_{ij}=|W_{ij}|\cdot \|X_j\|_2$  (Wanda의 핵심 직관)
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 015 · Code · def get_calib_dataset(tokenizer=None, n_samples=256, block_size=512): / dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation") / dataset = dataset.shuffle(seed=42)

- **현재 구간**: [실습 2] Calibration 데이터셋 준비
- **오른쪽에서 볼 코드**: `def get_calib_dataset(tokenizer=None, n_samples=256, block_size=512): / dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation") / dataset = dataset.shuffle(seed=42)`
- **정의되는 함수**: `get_calib_dataset`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 난수 seed를 고정한다. pruning threshold, data augmentation, optimizer 순서가 바뀌면 비교가 흔들리므로 baseline과 실험 조건을 고정하는 역할이다.
  - 데이터셋을 mini-batch 단위로 모델에 공급한다. CNN은 이미지 batch `[B,3,32,32]`, LLM은 token sequence `[B,T]`로 들어간다는 점을 구분한다.
  - calibration/observer는 activation range를 샘플 데이터로 추정한다. range가 좁으면 clipping, 넓으면 resolution 손실이 생긴다.
- **수학/shape 관점**:
  - $score_{ij}=|W_{ij}|\cdot \|X_j\|_2$  (Wanda의 핵심 직관)
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.
- **키워드**: 재현성, calibration, tokenizer

### Cell 016 · Markdown · 아래 빈칸을 작성하여 `activation_norm`을 계산하세요. Wanda 기법에서는 `activation_norm`을 L2 norm으로 계산하지만, 구현에서는 calibration 데이터셋에 대해 여러 번 반복하여 누적하므로 `activation_norm` 계산 시 제곱을 적용한 형태로 누적합니다. (이후 값을 사용할 때는 제곱근을 적용해야 합니다.)

- **현재 구간**: [실습 2] Calibration 데이터셋 준비
- **오른쪽에서 읽을 내용**: 아래 빈칸을 작성하여 `activation_norm`을 계산하세요. Wanda 기법에서는 `activation_norm`을 L2 norm으로 계산하지만, 구현에서는 calibration 데이터셋에 대해 여러 번 반복하여 누적하므로 `activation_norm` 계산 시 제곱을 적용한 형태로 누적합니다. (이후 값을 사용할 때는 제곱근을 적용해야 합니다.)
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - calibration/observer는 activation range를 샘플 데이터로 추정한다. range가 좁으면 clipping, 넓으면 resolution 손실이 생긴다.
  - Wanda는 weight magnitude만 보지 않고 입력 activation 크기도 곱한다. 같은 weight라도 자주/크게 활성화되는 channel에 연결되면 더 중요하다고 보는 방식이다.
- **수학/shape 관점**:
  - $score_{ij}=|W_{ij}|\cdot \|X_j\|_2$  (Wanda의 핵심 직관)
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 017 · Code · @torch.no_grad() / def get_calib_feat(model, tokenizer, samples): / input_dict = dict()

- **현재 구간**: [실습 2] Calibration 데이터셋 준비
- **오른쪽에서 볼 코드**: `@torch.no_grad() / def get_calib_feat(model, tokenizer, samples): / input_dict = dict()`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - quantization은 실수 범위를 정수 grid로 근사한다. `scale`은 grid 간격, `zero_point`는 실수 0이 대응되는 정수 위치다.
  - calibration/observer는 activation range를 샘플 데이터로 추정한다. range가 좁으면 clipping, 넓으면 resolution 손실이 생긴다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
  - quantization에서는 실수 tensor 자체보다 `(integer tensor, scale, zero_point/group_size)` 묶음이 실제 표현이다.
- **직접 구현할 때 체크**:
  - `YOUR CODE` 위치는 직접 구현 대상이다. 오른쪽 코드의 앞뒤 변수 이름과 반환 shape를 먼저 맞춘다.
  - 입력 tensor와 model parameter가 같은 device에 있는지 확인한다. CPU/GPU mismatch는 이 실습에서 흔한 오류다.
  - 평가/압축 적용 단계에서는 gradient를 끄는 이유를 확인한다. 메모리 절약과 accidental gradient 방지를 위한 장치다.
- **키워드**: zero-point, calibration, tokenizer, activation, Linear layer

### Cell 018 · Code · model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, use_safetensors=True) / input_feat = get_calib_feat(model, tokenizer, samples)

- **현재 구간**: [실습 2] Calibration 데이터셋 준비
- **오른쪽에서 볼 코드**: `model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, use_safetensors=True) / input_feat = get_calib_feat(model, tokenizer, samples)`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 학습/미세조정 루프다. forward → loss → backward → optimizer step 순서이며, pruning이면 step 이후 mask 유지, KD면 여러 loss 항의 비율을 확인한다.
  - calibration/observer는 activation range를 샘플 데이터로 추정한다. range가 좁으면 clipping, 넓으면 resolution 손실이 생긴다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
- **직접 구현할 때 체크**:
  - 입력 tensor와 model parameter가 같은 device에 있는지 확인한다. CPU/GPU mismatch는 이 실습에서 흔한 오류다.
- **키워드**: calibration, tokenizer

### Cell 019 · Markdown · [실습 3] Wanda Pruning 구현

- **현재 구간**: [실습 3] Wanda Pruning 구현
- **오른쪽에서 읽을 내용**: ## [실습 3] Wanda Pruning 구현 Wanda 논문을 참고하여 아래 빈 칸을 채워 Wanda Pruning을 구현하고 실행해보세요. Wanda에서는 일반적인 magnitude-based pruning 방법과 다르게게 weight parameter의 행(row)별로 동일한 희소성(sparsity)으로 pruning을 진행합니다.
- **무슨 작업인가**:
  - 이 셀은 뒤 코드가 왜 필요한지 알려주는 설명/문제 정의 셀이다. 오른쪽 노트북의 문장을 그냥 넘기지 말고, 바로 아래 코드 셀에서 어떤 변수·함수·측정값으로 구현되는지 연결해서 본다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - Wanda는 weight magnitude만 보지 않고 입력 activation 크기도 곱한다. 같은 weight라도 자주/크게 활성화되는 channel에 연결되면 더 중요하다고 보는 방식이다.
- **수학/shape 관점**:
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
- **직접 구현할 때 체크**:
  - 이 셀 실행 후 새로 생긴 변수 1~2개를 다음 셀이 어떻게 사용하는지 추적한다.

### Cell 020 · Code · @torch.no_grad() / def prune_wanda(model, sparsity, input_feat): / for n, m in model.named_modules():

- **현재 구간**: [실습 3] Wanda Pruning 구현
- **오른쪽에서 볼 코드**: `@torch.no_grad() / def prune_wanda(model, sparsity, input_feat): / for n, m in model.named_modules():`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - quantization은 실수 범위를 정수 grid로 근사한다. `scale`은 grid 간격, `zero_point`는 실수 0이 대응되는 정수 위치다.
  - Wanda는 weight magnitude만 보지 않고 입력 activation 크기도 곱한다. 같은 weight라도 자주/크게 활성화되는 channel에 연결되면 더 중요하다고 보는 방식이다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
  - quantization에서는 실수 tensor 자체보다 `(integer tensor, scale, zero_point/group_size)` 묶음이 실제 표현이다.
- **직접 구현할 때 체크**:
  - `YOUR CODE` 위치는 직접 구현 대상이다. 오른쪽 코드의 앞뒤 변수 이름과 반환 shape를 먼저 맞춘다.
  - 평가/압축 적용 단계에서는 gradient를 끄는 이유를 확인한다. 메모리 절약과 accidental gradient 방지를 위한 장치다.
- **키워드**: 희소도, 프루닝, 마스크, zero-point, Wanda, Linear layer

### Cell 021 · Code · Evaluate the model

- **현재 구간**: [실습 3] Wanda Pruning 구현
- **오른쪽에서 볼 코드**: `model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, use_safetensors=True) / model_perplexity = evaluate(model, tokenizer) / print(f"\nmodel perplexity (wanda 50%): {mode`
- **무슨 작업인가**:
  - 이 셀은 설명을 실제 PyTorch/Python 연산으로 바꾸는 실행 단위다. 실행 전후로 생성되는 변수, tensor shape, 모델 상태 변경 여부를 확인한다.
  - 모델 품질을 숫자로 고정하는 평가 루프다. CNN은 accuracy, LLM은 PPL을 주로 본다. 압축 실험에서는 압축률만큼이나 이 기준값의 변화량이 중요하다.
  - 학습/미세조정 루프다. forward → loss → backward → optimizer step 순서이며, pruning이면 step 이후 mask 유지, KD면 여러 loss 항의 비율을 확인한다.
  - pruning 구현의 핵심은 중요도 점수로 mask를 만들고 `W *= M`을 적용하는 것이다. 실제 속도 개선은 mask 형태가 hardware-friendly한지까지 봐야 한다.
  - Wanda는 weight magnitude만 보지 않고 입력 activation 크기도 곱한다. 같은 weight라도 자주/크게 활성화되는 channel에 연결되면 더 중요하다고 보는 방식이다.
- **수학/shape 관점**:
  - Linear layer는 보통 `Y=XW^T+b`로 동작한다. LLM에서 `X`는 `[B,T,d_in]`, `W`는 `[d_out,d_in]`로 보면 된다.
  - mask는 대상 weight와 같은 shape다. nonzero 비율은 `count_nonzero / numel`, sparsity는 `1 - nonzero_ratio`다.
  - PPL은 `exp(average cross entropy)`다. 낮을수록 다음 token 예측이 좋다.
- **직접 구현할 때 체크**:
  - 입력 tensor와 model parameter가 같은 device에 있는지 확인한다. CPU/GPU mismatch는 이 실습에서 흔한 오류다.
- **키워드**: 평가 루프, 희소도, 프루닝, tokenizer, perplexity, Wanda

## 3. 실습 후 스스로 확인할 질문

1. 이 노트북에서 baseline metric은 무엇이고, 압축 후 얼마만큼 변했는가?
2. in-place로 model weight를 바꾸는 셀은 어디이며, 원본 복구/reset은 어떻게 하는가?
3. 핵심 함수 하나를 빈 파일에 다시 구현한다면 입력/출력 shape를 주석으로 쓸 수 있는가?
4. 정확도/PPL 손실이 생겼을 때 원인이 range 추정, mask 단위, scale 선택, calibration 부족 중 어디에 가까운가?
5. 실제 hardware speedup으로 이어지려면 단순 parameter 감소 외에 어떤 조건이 필요한가?
