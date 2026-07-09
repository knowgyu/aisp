# RAG

## BM25가 TF-IDF 이후 등장한 이유

BM25는 `df=0` 문제 때문에 등장한 것이 아니다.

TF-IDF의 주요 한계:

* TF(term frequency)가 선형 증가한다.

  * 단어가 1번 → 2번 등장하는 차이와
  * 100번 → 101번 등장하는 차이를 동일하게 취급한다.
  * 하지만 실제 검색에서는 반복 등장할수록 추가 정보량은 감소한다.

* 긴 문서 문제

  * 긴 문서는 단순히 단어가 많이 포함될 가능성이 높다.
  * 문서 길이를 고려하지 않으면 긴 문서가 과도하게 높은 점수를 받을 수 있다.

BM25의 개선:

* TF saturation

  * 단어가 많이 등장할수록 점수 증가량을 감소시킨다.
  * 처음 등장하는 몇 번은 중요하지만, 100번 반복되는 것은 추가 의미가 적다고 본다.

* Document length normalization

  * 긴 문서와 짧은 문서를 보정한다.

* IDF 개선

  * smoothing을 통해 극단적인 값을 완화한다.

---

## BM25의 k1

BM25의 TF 부분:

[
\frac{TF(k_1+1)}{TF+k_1}
]

`k1`은 TF saturation 정도를 조절하는 파라미터이다.

* k1이 크면:

  * TF 증가를 더 오래 인정
  * TF-IDF와 비슷하게 동작

* k1이 작으면:

  * 빠르게 saturation
  * 단어 존재 여부를 더 중요하게 봄

일반적인 값:

* k1 ≈ 1.2 ~ 2.0

---

## BM25의 smoothing

BM25 IDF:

[
IDF=\log\frac{N-df+0.5}{df+0.5}
]

`+0.5` 같은 smoothing term은:

* df=0 상황에서 문제 방지
* 극단적인 IDF 값 완화

목적이다.

하지만 BM25의 핵심 등장 이유는:

* TF saturation
* document length normalization

이다.

---

# Ranking Evaluation

## Ranking Task

입력:

```
Query
```

출력:

```
Ranked list of documents
```

예:

```
1. Document A
2. Document B
3. Document C
...
```

평가를 위해서는 해당 query에 대한 정답 집합이 필요하다.

```
Relevant documents:
{D2, D5, D7}
```

처럼 정의된다.

---

# MRR (Mean Reciprocal Rank)

목적:

> 정답 문서가 얼마나 빨리 등장하는지 평가

첫 번째 relevant document의 rank만 사용한다.

[
RR=\frac{1}{rank}
]

예:

```
1. irrelevant
2. relevant
3. relevant
```

이면:

[
RR=\frac{1}{2}
]

여러 query에 대해 평균:

[
MRR=\frac{RR_1+RR_2+...+RR_n}{n}
]

특징:

* relevant document가 여러 개 있어도 첫 번째 정답 위치만 고려
* "정답 하나를 빨리 찾는 것"이 중요한 task에 적합

예:

모든 query에서 정답이 1등이면:

[
MRR=1
]

---

# Relevance Label은 어떻게 정하는가?

검색 모델이 직접 아는 것이 아니다.

평가 데이터에 사람이 만든 relevance judgment가 존재한다.

예:

```
Query:
BM25란?

Document:
BM25는 TF-IDF를 개선한 ranking function이다.

Label:
Relevant
```

생성 방법:

* 사람 annotation
* 전문가 annotation
* benchmark dataset 제공

대표적인 형태:

```
query_id
document_id
relevance_score
```

---

# Precision@K / Recall@K

## Precision@K

질문:

> 내가 가져온 K개의 문서 중 얼마나 맞았는가?

공식:

[
Precision@K=
\frac{TopK\ 중\ relevant\ 개수}{K}
]

예:

Top 5:

```
정답
오답
오답
정답
오답
```

이면:

[
Precision@5=\frac{2}{5}
]

---

## Recall@K

질문:

> 전체 정답 문서 중 얼마나 찾아왔는가?

공식:

[
Recall@K=
\frac{TopK에서\ 찾은\ relevant\ 개수}
{전체\ relevant\ 개수}
]

예:

전체 relevant 문서:

```
40개
```

Top 5 결과에서:

```
2개 발견
```

이면:

[
Precision@5=\frac25=0.4
]

[
Recall@5=\frac2{40}=0.05
]

Precision은 가져온 결과의 품질,
Recall은 전체 정답 회수율을 의미한다.

---

# K 선택

K가 작으면:

* Precision 증가 가능
* Recall 감소 가능

K가 크면:

* Recall 증가
* irrelevant 문서 증가 가능

F1:

[
F1=2\frac{Precision \times Recall}{Precision+Recall}
]

을 이용해 균형점을 찾을 수 있다.

하지만 실제 검색에서는 목적에 따라 metric을 선택한다.

* Web Search:

  * MRR
  * NDCG@K
  * Precision@K

* RAG:

  * Recall@K
  * Context quality

* 의료/법률 검색:

  * Recall 중요

---

# LLM을 Relevance Judge로 사용할 수 있는가?

가능하다.

기존 방식:

```
Query + Document
        ↓
Human Annotation
        ↓
Relevant Label
```

최근 방식:

```
Query + Document
        ↓
LLM Judge
        ↓
Relevance Score
```

가능한 출력:

Binary:

```
Relevant / Not Relevant
```

Graded relevance:

```
0: irrelevant
1: 조금 관련
2: 관련
3: 매우 관련
```

장점:

* 빠른 평가
* 대규모 평가 가능

문제:

* LLM bias
* Position bias
* 비용

따라서 benchmark의 gold label은 아직 사람 annotation이 많이 사용된다.

---

# GPT Judge와 LLaMA Judge 차이

차이는 encoder/decoder 구조 차이가 아니다.

둘 다:

* Decoder-only Transformer
* Causal self-attention

구조이다.

차이는:

* 학습 데이터
* Instruction tuning
* Alignment 방식
* 모델 크기

등이다.

Judge 역할:

```
Query
+
Document
+
Evaluation Prompt

↓

LLM

↓

Relevance Score
```

---

# Encoder의 출력

Encoder 출력은:

> 의미 정보를 담은 고차원 tensor

이다.

예:

입력:

```
나는 학교에 간다
```

Encoder 출력:

```
[
 vector1,
 vector2,
 vector3,
 ...
]
```

형태.

사람이 읽는 의미가 아니라:

* 의미 정보가 압축된 숫자 표현
* hidden representation

이다.

---

# BERT vs BART

## BERT

구조:

```
Encoder-only
```

목적:

* 문장 이해
* representation 생성

학습:

* Masked Language Modeling

강점:

* Classification
* Retrieval
* Reranking
* Relevance 판단

---

## BART

구조:

```
Encoder + Decoder
```

목적:

* 입력 이해
* 출력 생성

학습:

* Noise가 추가된 문장을 원래 문장으로 복원

강점:

* Translation
* Summarization
* Text generation

---

# BM25 vs Dense Retrieval

## BM25

방식:

> Keyword matching 기반 검색

장점:

* 정확한 단어 검색
* 숫자
* 이름
* 코드
* ID

에 강함.

단점:

* 표현이 바뀌면 약함

예:

```
자동차
```

와

```
차량
```

은 의미가 비슷하지만 keyword는 다르다.

---

## Dense Retrieval

방식:

Query와 Document를 embedding vector로 변환 후 similarity 계산.

```
Query
 ↓
Encoder
 ↓
Vector

Document
 ↓
Encoder
 ↓
Vector

↓

Similarity 계산
```

장점:

* 의미 기반 검색
* Paraphrase 대응

단점:

* 숫자
* Parameter table
* Exact value

에 약할 수 있다.

예:

```
GPU A:
Power 320W

GPU B:
Power 450W

GPU C:
Power 600W
```

Query:

```
GPU B power?
```

Dense retrieval은:

"GPU power 정보"

라는 의미는 잘 찾지만,

"450W와 600W의 차이"

같은 정확한 값 구분은 놓칠 수 있다.

---

# 실제 검색 시스템 구조

현대 검색 시스템은 보통 하나만 사용하지 않는다.

```
전체 문서

↓

BM25 + Dense Retrieval
(후보 생성)

↓

Top 100~1000

↓

Reranker
(BERT Cross Encoder / LLM)

↓

Top K 반환
```

역할:

* BM25:

  * 정확한 keyword matching
  * Exact search

* Dense Retrieval:

  * 의미 검색
  * 표현 변화 대응

* Reranker:

  * Query와 Document를 직접 비교
  * 최종 relevance 판단

---

# 최종 정리

검색 시스템은 다음 역할 분담으로 이해하면 된다.

```
BM25
= 단어 기반 검색

Dense Retrieval
= 의미 기반 검색

Reranker
= 정밀한 relevance 판단

MRR
= 첫 정답 위치 평가

Precision@K
= 가져온 결과의 정확도

Recall@K
= 전체 정답 회수율
```

좋은 검색 시스템은 보통:

```
BM25 + Dense Retrieval + Reranker
```

조합으로 구성된다.

---

search도 retrieval인데,

고차원 데이터는 인덱싱해봤자 성능 안 나옴(O(n)이라). FBERT(q) -> 512dim, 1024dim, ... 
-> Curse of dimensionality


---

````md
# RAG 실습 및 심화 내용 정리

> IPython Notebook를 보면서 생긴 궁금증들과, 그에 대한 핵심 정리

---

# 1. 이 실습은 어떤 RAG인가?

처음에는

> "질문을 하면 Wikipedia를 실시간으로 검색해서 가져오는 건가?"

라고 생각했는데, 실제 코드를 보면 그렇지 않았다.

```python
with open(city_name_path, 'r', encoding='utf-8') as file:
    ...
    city_names.append(city)

reader = WikipediaReader()
documents = reader.load_data(city_names)

index = VectorStoreIndex.from_documents(documents)
```

즉,

```
city_short.txt
        │
        ▼
도시 이름 목록

        │
        ▼
Wikipedia에서 미리 읽음

        │
        ▼
Document 생성

        │
        ▼
Embedding

        │
        ▼
Vector DB(Index)
```

즉 **Offline Indexing** 방식이다.

질문이 들어올 때마다 Wikipedia를 검색하는 것이 아니라,

미리 만들어 놓은 Vector DB에서 Retrieval을 수행한다.

---

# 2. 전체 RAG 흐름

실습의 핵심 흐름은 다음과 같다.

```
User Query

      │

      ▼

Retriever
(Vector Search)

      │

Top-k Chunks

      │

      ▼

Generator
(LLM)

      │

      ▼

Final Answer
```

이후 실습에서는

- chunk size
- overlap
- top-k
- prompt

등을 변경하면서 결과를 비교한다.

---

# 3. Chunk Size란?

Chunk Size는

문서를 얼마만큼 잘라서 Vector DB에 저장할지를 의미한다.

예를 들어

```
Document

↓

200 token씩

↓

chunk1
chunk2
chunk3
...
```

또는

```
Document

↓

1000 token씩

↓

chunk1
chunk2
...
```

### Chunk가 크면

장점

- 문맥 유지
- 긴 설명이 하나의 Chunk 안에 있음

단점

- 불필요한 정보까지 함께 들어감
- Context가 커짐

---

### Chunk가 작으면

장점

- 검색 정확도 증가
- 필요한 정보만 가져오기 쉬움

단점

- 문맥이 끊길 수 있음

---

# 4. Chunk Overlap

예를 들어

chunk size = 1000

overlap = 200

이면

```
chunk1

0~1000

chunk2

800~1800

chunk3

1600~2600
```

처럼 일부를 겹쳐 저장한다.

목적은

문장이 Chunk 경계에서 잘리는 것을 방지하기 위함이다.

---

# 5. Top-k

Retriever가

가장 유사한 Chunk를 몇 개 가져올지 결정한다.

예를 들어

```
top_k=3

↓

chunk5

chunk120

chunk802
```

를 가져온다.

---

# 6. Chunk Size × Top-k

예를 들어

```
chunk size = 2000

top_k = 10
```

이면

대략

```
2000 token

×

10 chunks

≈

20000 token
```

정도의 Context가 LLM으로 전달된다.

(실제 token 수는 조금 달라질 수 있음)

---

# 7. Context Window보다 Retrieval 결과가 더 크면?

예를 들어

```
Model Context Window

64K

Retrieval 결과

80K
```

라면

모델은 80K를 모두 받을 수 없다.

보통 다음 중 하나가 발생한다.

### 1.

입력 길이 초과

API Error

---

### 2.

Framework가 자동으로 잘라냄

뒤쪽 Context가 사라질 수도 있다.

---

### 3.

Compression

Summary 등을 통해 Context를 줄인다.

---

따라서

Chunk Size와 Top-k를 무작정 크게 하는 것은 좋지 않다.

---

# 8. Attention Dilution

질문

> "의미 없는 정보도 함께 들어오면 Attention이 분산되는 건가?"

답은 그렇다.

예를 들어

```
Question

↓

Relevant Chunk

↓

Noise Chunk

↓

Noise Chunk

↓

Noise Chunk
```

가 들어오면

Self-Attention은

모든 Token을 서로 참고한다.

즉

```
Question

↓

Relevant Token

↓

불필요한 Token들
```

까지 모두 Attention 계산 대상이 된다.

그래서

중요 정보가 상대적으로 묻히는 현상이 발생한다.

이를

**Attention Dilution**

이라고 한다.

즉

Context가 길다고 무조건 좋은 것이 아니다.

---

# 9. 중요한 Chunk 하나만 Retrieval된 경우

예를 들어

```
chunk1

...

chunk552

chunk553

chunk554

chunk555  ← Retrieval

chunk556

chunk557

chunk558
```

Top-k 결과가

```
chunk555
```

뿐이었다.

그런데

실제로는

```
552~558
```

을 모두 봐야 답할 수 있는 상황이라면?

---

## 단순 해결법

Window Expansion

```
retrieve

↓

chunk555

↓

자동으로

552~558 추가
```

하지만

필요 없는 Chunk도 같이 들어올 수 있다.

---

# 10. Expansion의 문제점

Retriever 입장에서는

```
chunk555
```

가 중요한지는 알지만

주변 Chunk가 필요한지는 모른다.

괜히

```
chunk552

chunk553

chunk554
```

를 추가하면

Context만 길어진다.

즉

Recall은 증가하지만

Precision은 감소할 수 있다.

---

# 11. 실제 시스템에서는 어떻게 해결하는가?

## (1) Parent-Child Retrieval

```
Parent

500~600

        │

 ├── child551

 ├── child552

 ├── child553

 ├── child555
```

검색은

Child로 하고

최종적으로

Parent를 가져온다.

---

## (2) Heading-aware Retrieval

Markdown 예시

```
# Product

## Price

chunk1

chunk2

## Specification

chunk3

chunk4

chunk5
```

chunk4가 검색되면

같은 Heading 아래

```
chunk3

chunk4

chunk5
```

를 함께 가져온다.

---

## (3) Expansion → Reranking

```
Retriever

↓

후보 30개

↓

Reranker

↓

최종 5개
```

Recall과 Precision을 동시에 확보하는 방식이다.

---

## (4) Context Compression

많이 가져온 뒤

```
10000 token

↓

Summary

↓

1000 token
```

으로 줄인다.

---

# 12. Cosine Similarity의 한계

Retriever는

```
Question

↓

Cosine Similarity
```

만 보고 검색한다.

하지만

```
Similarity
≠

Answer에 필요한 정보
```

이다.

예를 들어

질문

```
왜 상대성이론이 등장했는가?
```

Retriever는

```
상대성이론 정의
```

를 가져올 수 있다.

하지만 실제로는

```
당시 물리학의 한계
```

가 더 중요한 정보일 수 있다.

즉

Semantic Similarity와

Answer Utility는 다르다.

---

# 13. 그렇다면 Query 자체를 여러 개 만들면?

질문

> "LLM이 필요한 정보를 먼저 분석하고 여러 Query를 만들어 Retrieval하면 더 좋지 않을까?"

실제로 그렇다.

이를

- Query Transformation
- Query Expansion
- Multi-Query Retrieval
- Query Decomposition

등이라고 한다.

---

예시

원래 질문

```
상대성이론의 배경과 영향을 설명해줘
```

↓

LLM이 분해

```
Q1

상대성이론 등장 배경

Q2

핵심 내용

Q3

현대 물리학 영향
```

↓

각각 Retrieval

↓

합쳐서 Answer 생성

---

# 14. HyDE

질문 대신

LLM이

"이상적인 답변"

을 먼저 작성한다.

```
Question

↓

Hypothetical Answer

↓

Embedding

↓

Retrieval
```

실제 문서 형태와 더 비슷하기 때문에

검색 성능이 좋아질 수 있다.

---

# 15. Query Decomposition도 완벽하지 않다.

단점

### 1.

LLM이

필요한 정보를 잘못 판단할 수 있다.

---

### 2.

Retrieval 횟수가 증가한다.

```
Query

↓

5개의 Sub Query

↓

Retrieval ×5
```

Latency 증가

비용 증가

---

### 3.

너무 많은 Query는

후보 Chunk를 지나치게 많이 가져온다.

결국

다시 Filtering이 필요하다.

---

# 16. 현대 RAG 구조

실제 Production에서는

보통

```
User Question

        │

        ▼

Query Analysis

        │

        ▼

Multi Query

        │

        ▼

Retriever

        │

        ▼

Candidate Chunks

        │

        ▼

Reranker

        │

        ▼

Context Expansion

        │

        ▼

Compression

        │

        ▼

LLM

        │

        ▼

Answer
```

처럼 여러 단계를 거친다.

단순히

```
Top-k

Chunk Size
```

만 조절하는 것이 아니라,

검색 전략 자체를 최적화하는 것이 현대 RAG의 핵심이다.

---

# 이번 실습의 수준

이 Notebook은

Production 수준의 RAG 구현이 아니라,

**RAG의 기본 흐름을 이해하기 위한 교육용 실습**이다.

다루는 내용

- OpenAI API 호출
- Wikipedia 문서 로드
- Vector Index 생성
- Retriever
- Query Engine
- Chunk Size 변경
- Top-k 변경
- Prompt 변경

반면 실제 서비스에서 자주 사용하는

- Hybrid Search
- BM25 + Dense Retrieval
- Parent-Child Retrieval
- Metadata Filtering
- Query Decomposition
- Multi-Query Retrieval
- Reranker
- Context Compression
- Agentic RAG

등은 포함되어 있지 않다.
````
