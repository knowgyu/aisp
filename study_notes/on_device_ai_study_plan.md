# On-Device AI 시험 대비 학습 준비 문맥

> 상태: 시작 전 준비용. 실제 챕터 학습은 아직 시작하지 않는다.  
> 자료 위치: `On-Device AI 강의자료/`  
> 원칙: 각 챕터마다 **슬라이드/PDF로 큰 그림 → 빈칸 ipynb 직접 읽기 → 코드 한 줄씩 해석 → answer ipynb로 검산 → 시험형 요약** 순서로 진행한다.

## 0. 전체 자료 맵

### 공통 강의 자료
- `ODAI-1.pdf`
- `ODAI-2.pdf`
- `실습/AVD.Intro.ET.pdf`
- `실습/environment_env_aias_test.yml`

### 실습 챕터
1. `1. Pruning for CNN.ipynb`
   - 정답: `1. Pruning for CNN_answer.ipynb`
   - 슬라이드: `1. Pruning for CNN_slide.pdf`
2. `2. Quantization for CNN.ipynb`
   - 정답: `2. Quantization for CNN_answer.ipynb`
   - Colab: `2. Quantization for CNN_colab.ipynb`, `2. Quantization for CNN_answer_colab.ipynb`
   - 슬라이드: `2. Quantization for CNN_slide.pdf`, `2. Quantization for CNN_slide.docx`
3. `3. Knowledge Distillation.ipynb`
   - 정답: `3. Knowledge Distillation_answer.ipynb`
   - 슬라이드: `3. Knowledge Distillation_slide.pdf`
4. `4. Pruning for LLM.ipynb`
   - 정답: `4. Pruning for LLM_answer.ipynb`
   - 슬라이드: `4. Pruning for LLM_slide.pdf`
5. `5. Quantization for LLM.ipynb`
   - 정답: `5. Quantization for LLM_answer.ipynb`
   - Colab: `5. Quantization for LLM_colab.ipynb`, `5. Quantization for LLM_colab_answer.ipynb`
   - 슬라이드: `5. Quantization for LLM_slide.pdf`

## 1. 학습 운영 방식

각 챕터는 같은 템플릿으로 진행한다.

### Phase A — 챕터 오리엔테이션
목표: “이 챕터가 왜 필요한지”를 먼저 잡는다.
- 슬라이드/PDF 목차 확인
- 실습 notebook의 Goals/Contents 확인
- 핵심 문제 정의: 메모리, 연산량, latency, bandwidth, 정확도 저하 중 무엇을 줄이려는가?
- 시험 대비 키워드 10개 추출

### Phase B — 개념 뼈대 정리
목표: 코드 전에 수식과 용어를 이해한다.
- 핵심 알고리즘을 말로 설명
- 입력/출력 tensor shape 정리
- 어떤 metric으로 성능을 평가하는지 정리
- trade-off: 압축률 vs 정확도, 구조적 sparsity vs 비구조적 sparsity, PTQ vs QAT 등

### Phase C — ipynb 코드 정독
목표: 코드 한 셀씩 “왜 이 줄이 필요한지”까지 본다.
- 빈칸 notebook 기준으로 진행
- import/setup은 대충 넘기지 않고, dataset/model/eval 함수 역할을 구분
- 함수마다 다음 4개를 기록
  1. 입력
  2. 출력
  3. 내부 수식/연산
  4. 실험에서 쓰이는 위치

### Phase D — answer ipynb 검산
목표: 정답을 암기하지 않고, 내 추론과 비교한다.
- 먼저 직접 빈칸의 의도를 설명
- answer에서 구현 차이 확인
- 틀린 부분은 “내가 놓친 전제”로 기록

### Phase E — 시험형 압축
목표: 시험 직전 다시 볼 수 있는 형태로 만든다.
- 한 장 요약
- 핵심 수식
- 코드 패턴
- 자주 헷갈리는 포인트
- 예상 문제와 답변

## 2. 챕터별 페이즈

## Chapter 1 — CNN Pruning

### 핵심 질문
CNN에서 중요하지 않은 weight/filter/channel을 제거할 때, 정확도를 얼마나 유지하면서 모델 크기와 연산량을 줄일 수 있는가?

### 주요 주제
- CIFAR-10 / VGG 기반 실험 환경
- 모델 크기와 정확도 평가
- weight distribution 확인
- reconstruction error
- pruning granularity/pattern
  - fine-grained
  - vector-level
  - kernel-level
  - channel-level
- pruning ratio
  - layer-wise sensitivity
  - global magnitude pruning
- pruning schedule
  - one-shot pruning
  - iterative pruning
  - linear / cubic sparsity scheduler

### 코드 학습 포인트
- mask 생성 방식
- threshold 계산 방식
- pruning ratio를 layer별/global로 적용하는 차이
- pruned weight가 실제 연산/정확도에 미치는 영향
- fine-tuning loop에서 mask가 유지되는 방식

### 시험 대비 포인트
- structured vs unstructured pruning 차이
- sparsity가 높아질수록 accuracy가 떨어지는 이유
- global pruning이 layer-wise pruning과 다른 이유
- reconstruction error가 의미하는 것

## Chapter 2 — CNN Quantization

### 핵심 질문
CNN의 FP32 연산/저장을 INT8 등 낮은 bit-width로 바꾸면 모델 크기와 추론 비용을 얼마나 줄일 수 있는가?

### 주요 주제
- n-bit integer 표현
- linear quantization
  - scale
  - zero point
  - clipping / rounding
- per-tensor vs per-channel quantization
- quantized inference
  - quantized fully-connected layer
  - quantized convolution layer
- non-uniform quantization
- K-means quantization
- quantization-aware training(QAT)
- PyTorch quantization API

### 코드 학습 포인트
- `q = round(r / S) + Z` 형태의 변환
- dequantization과 pseudo quantization의 의미
- bias quantization이 왜 별도 scale을 가지는지
- activation scale과 weight scale이 output scale로 연결되는 방식
- quantized conv/fc에서 누산 accumulator dtype이 중요한 이유

### 시험 대비 포인트
- scale/zero point 계산 유도
- symmetric vs asymmetric quantization
- PTQ와 QAT 차이
- per-channel quantization이 weight에 유리한 이유
- K-means quantization이 linear quantization과 다른 이유

## Chapter 3 — Knowledge Distillation

### 핵심 질문
큰 teacher 모델의 지식을 작은 student 모델로 옮겨, 작은 모델의 정확도를 끌어올릴 수 있는가?

### 주요 주제
- CIFAR-10 data loading
- VGG teacher / lightweight student
- baseline CE 학습
- KD loss
  - hard label cross entropy
  - soft target / KL divergence
  - temperature
- cosine similarity 기반 KD
- intermediate regressor / hint-based KD
- MSE loss로 feature map 정렬

### 코드 학습 포인트
- teacher는 eval/frozen 상태인지 확인
- student만 update되는지 확인
- logits distillation과 feature distillation의 차이
- temperature가 soft label 분포에 미치는 영향
- regressor가 feature channel/shape mismatch를 맞추는 방식

### 시험 대비 포인트
- teacher/student 구조 설명
- hard label vs soft label
- temperature를 쓰는 이유
- feature-level KD가 logits KD와 다른 점
- distillation이 compression 방법인 이유

## Chapter 4 — LLM Pruning

### 핵심 질문
LLM의 weight를 제거해도 perplexity를 크게 망치지 않으려면 어떤 기준으로 pruning해야 하는가?

### 주요 주제
- SmolLM-135M 모델 로딩
- perplexity 기반 평가
- magnitude-based pruning
- calibration dataset
- activation norm 계산
- Wanda pruning
  - weight magnitude + activation statistics

### 코드 학습 포인트
- LLM에서 pruning 대상이 되는 linear layer 찾기
- pruning mask를 만드는 기준
- calibration input을 forward hook 등으로 수집하는 방식
- activation norm이 weight importance와 결합되는 방식
- perplexity 계산 흐름

### 시험 대비 포인트
- CNN pruning과 LLM pruning의 차이
- perplexity가 낮을수록 좋은 이유
- magnitude-only pruning의 한계
- Wanda가 activation 정보를 쓰는 이유
- calibration dataset의 역할

## Chapter 5 — LLM Quantization

### 핵심 질문
LLM을 낮은 bit-width로 줄일 때, outlier channel과 activation/weight quantization difficulty를 어떻게 다룰 것인가?

### 주요 주제
- FP16 LLM 메모리 부담
- weight-only quantization
- AWQ
  - salient channels
  - scale factor search
- pseudo quantization
- weight-activation quantization
- SmoothQuant류 difficulty migration
- scale factor sampling
- rotation-based quantization
  - QuaRot / SpinQuant 아이디어
- LayerNorm ↔ Linear fusion

### 코드 학습 포인트
- weight-only quantization과 W/A quantization의 연산 차이
- pseudo quantization이 실제 int kernel 없이 효과를 시뮬레이션하는 방식
- salient channel을 보호하기 위해 scale을 조정하는 이유
- SmoothQuant에서 activation 난이도를 weight로 옮기는 방식
- rotation matrix가 quantization outlier를 완화하는 직관

### 시험 대비 포인트
- W4A16, W8A8 같은 표기 의미
- AWQ와 SmoothQuant의 목적 차이
- outlier channel이 quantization에 치명적인 이유
- LayerNorm fusion이 필요한 이유
- rotation 기반 quantization의 수식 직관

## 3. 권장 진행 순서

1. **전체 오리엔테이션**
   - `ODAI-1.pdf`, `ODAI-2.pdf`를 훑고 전체 그림을 잡는다.
   - 목적: On-device 제약 = memory, compute, bandwidth, latency, energy.

2. **CNN 압축 파트**
   - Chapter 1 Pruning CNN
   - Chapter 2 Quantization CNN
   - 이유: 작은 모델/이미지 태스크에서 pruning/quantization 기본기를 먼저 잡는다.

3. **지식 증류 파트**
   - Chapter 3 Knowledge Distillation
   - 이유: pruning/quantization과 달리 “작은 모델을 학습으로 좋게 만드는” 압축 방법을 분리해서 이해한다.

4. **LLM 압축 파트**
   - Chapter 4 Pruning LLM
   - Chapter 5 Quantization LLM
   - 이유: CNN에서 배운 compression 개념이 LLM에서는 activation outlier, perplexity, calibration, memory bandwidth 문제로 확장된다.

5. **시험 직전 통합 비교**
   - Pruning vs Quantization vs Distillation
   - CNN vs LLM
   - PTQ vs QAT
   - weight-only vs weight-activation quantization
   - magnitude-only vs activation-aware methods

## 4. 매 챕터 산출물 템플릿

각 챕터를 공부할 때 아래 파일을 새로 만든다.

```text
study_notes/on_device_ai/
  01_cnn_pruning_deep_review.md
  02_cnn_quantization_deep_review.md
  03_knowledge_distillation_deep_review.md
  04_llm_pruning_deep_review.md
  05_llm_quantization_deep_review.md
```

각 파일 구조:

```md
# Chapter N. 제목

## 1. 이 챕터의 목적
## 2. 핵심 용어
## 3. 핵심 수식
## 4. 코드 셀별 해설
## 5. 실습 빈칸 의도와 정답 비교
## 6. 실험 결과 해석
## 7. 시험 예상 질문
## 8. 내가 헷갈린 점
```

## 5. 첫 시작 시 바로 할 일

다음 턴에서 실제 공부를 시작하면 아래 순서로 들어간다.

1. `ODAI-1.pdf`, `ODAI-2.pdf`의 목차/큰 그림 확인
2. `1. Pruning for CNN_slide.pdf`로 Chapter 1 개념 워밍업
3. `1. Pruning for CNN.ipynb`를 셀 0부터 순서대로 정독
4. 빈칸/구현부는 바로 answer를 보지 않고 먼저 의도 설명
5. 이후 `1. Pruning for CNN_answer.ipynb`로 검산
6. `study_notes/on_device_ai/01_cnn_pruning_deep_review.md`에 시험 대비 노트 생성

## 6. 학습 중 지켜야 할 규칙

- answer notebook은 “정답지”가 아니라 “검산용”으로만 본다.
- 코드 설명은 항상 shape, dtype, device, gradient 흐름을 같이 본다.
- 실험 결과는 숫자만 보지 말고 trade-off로 해석한다.
- 시험 대비는 암기보다 비교표 중심으로 정리한다.
- CNN에서 배운 개념을 LLM에서 어떻게 변형하는지 계속 연결한다.
