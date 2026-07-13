# RAG Day 1 실습 코드 동반 노트: LlamaIndex Query Engine

> 대상 원본: `rag/1일차/실습 자료/Code/1. Llama_index.ipynb`, `rag/1일차/실습 자료/Code/2. RAG.ipynb`  
> 목적: LlamaIndex의 객체 책임과 데이터 흐름을 코드 순서대로 복원하고, 시험에서 shape·ID·검색 지표를 설명한다.

## 1. 한 줄 지도

```text
파일/문서 -> Document -> Node(chunk + metadata) -> embedding(vector)
       -> VectorStoreIndex -> Retriever(top-k NodeWithScore)
       -> ResponseSynthesizer/LLM -> Response(source_nodes)
```

LlamaIndex는 LLM 자체가 아니라 **문서 적재, chunking, 색인, 검색, 답변 합성**을 연결하는 orchestration 계층이다. 검색 결과를 만들기 전에는 생성 모델을 호출하지 않는다.

## 2. 핵심 객체와 책임

| 객체/API | 책임 | 시험에서 구분할 점 |
|---|---|---|
| `SimpleDirectoryReader` | 디렉터리의 파일을 읽어 `Document` 목록 생성 | 파일 I/O와 parsing 담당, 검색하지 않음 |
| `Document` | 원문 text와 `metadata`를 담는 입력 단위 | 아직 검색용 chunk가 아님 |
| `SentenceSplitter`/`NodeParser` | 문서를 작은 `Node`로 분할 | `chunk_size`, `chunk_overlap`이 recall/context에 영향 |
| `Node` | 검색·임베딩되는 chunk. `node_id`와 metadata 보존 | 답변 source citation의 최소 단위 |
| `Embedding` | text를 고정 길이 벡터로 변환 | 벡터 차원 `d`는 모델 설정에 따름 |
| `VectorStoreIndex` | Node와 embedding을 저장하고 검색기 생성 | index 생성 자체가 답변 생성은 아님 |
| `as_retriever`/`Retriever` | 질문을 벡터화해 유사 Node 후보 반환 | `similarity_top_k=k`는 후보 수 |
| `NodeWithScore` | Node + 유사도 score 묶음 | score는 답변 정답 확률이 아님 |
| `get_response_synthesizer` | 검색 context를 prompt에 넣어 응답 합성 | retrieval과 generation을 분리 평가 |
| `QueryEngine` | retriever → synthesizer를 한 번에 호출 | 사용자가 호출하는 편의 인터페이스 |
| `Response` | 답변 text와 `source_nodes`를 함께 보관 | 근거 문서와 score를 확인할 수 있음 |

## 3. 코드 실행 순서

### 3.1 문서 적재와 Node 생성

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

raw_docs = SimpleDirectoryReader(input_dir="data").load_data()
parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = parser.get_nodes_from_documents(raw_docs)
```

- `raw_docs: list[Document]`: 문서 수를 `N_doc`이라 하면 각 원소는 길이가 서로 다른 문자열이다.
- `nodes: list[BaseNode]`: chunk 수를 `N_node`라 하면 `N_node >= N_doc`가 일반적이다.
- Node에는 `text`, `metadata`, `node_id`가 있다. 한 문서가 여러 Node로 쪼개져도 원본 파일 metadata를 이어 받아 출처를 추적한다.
- `chunk_overlap`은 인접 chunk 사이의 반복 토큰이다. 너무 작으면 경계 의미가 끊기고, 너무 크면 저장·검색 비용이 증가한다.

### 3.2 임베딩과 VectorStoreIndex

```python
index = VectorStoreIndex(nodes, embed_model=embed_model)
retriever = index.as_retriever(similarity_top_k=3)
```

Node의 텍스트를 embedding model로 바꾼다.

```text
node text_i -> embedding_i ∈ R^d
전체 색인 -> matrix E ∈ R^(N_node × d)
query q -> embedding(q) ∈ R^d
```

`d`는 embedding 모델 차원이며 임의로 `768`로 가정하면 안 된다. cosine similarity의 전형적인 형태는 다음과 같다.

\[
\operatorname{sim}(q, x_i)=\frac{q\cdot x_i}{\|q\|\|x_i\|}
\]

검색기는 score가 큰 Node를 `k`개 반환한다. 반환은 보통 `list[NodeWithScore]`이며 각 원소의 `node`와 `score`를 함께 확인한다.

## 4. Query Engine과 context data flow

```python
query_engine = index.as_query_engine(
    similarity_top_k=3,
    response_mode="compact",
)
response = query_engine.query("RAG의 장점은 무엇인가?")
```

내부 흐름은 다음과 같다.

1. 질문 문자열 `q`를 embedding하여 `R^d` query vector로 만든다.
2. Vector store에서 similarity top-k Node를 고른다.
3. Node text와 metadata를 context로 합친다.
4. query + context를 LLM prompt에 넣는다.
5. `Response.response`에 생성 text를, `Response.source_nodes`에 근거를 보관한다.

context가 `k`개 chunk라면 대략 입력은 `[question] + [context_1, ..., context_k]`다. 각 chunk 길이가 `L` token이면 context는 최악의 경우 `O(kL)` token이므로, `k`를 늘리면 recall은 좋아질 수 있어도 context limit·지연·잡음이 증가한다.

## 5. Node ID와 vector ID를 혼동하지 않기

- **Node ID (`node_id`)**: LlamaIndex가 chunk를 식별하는 논리 ID. metadata와 source citation을 연결한다.
- **Vector ID**: vector store가 embedding row를 식별하는 저장소 키. 구현에 따라 Node ID를 그대로 쓰거나 별도 키를 둘 수 있다.
- **Document ID**: 원본 파일/문서를 식별하는 상위 출처. 하나의 Document에 여러 Node가 매핑된다.

따라서 “벡터 하나 = 문서 하나”가 아니다. 보통은 `1 Document -> many Nodes -> many vectors`다. 답변 근거를 표시할 때는 `source_nodes[i].node.metadata`와 `node_id`를 사용하고, 유사도 score를 정답 확률처럼 해석하지 않는다.

## 6. LlamaIndex Query Engine과 직접 조립한 Custom Engine

기본 `index.as_query_engine()`은 빠른 구성에 적합하다. 실습의 RAG 앱에서는 다음처럼 책임을 직접 조립하는 패턴도 중요하다.

```python
retriever = index.as_retriever(similarity_top_k=top_k)
synthesizer = get_response_synthesizer(
    response_mode="compact",
    text_qa_template=qa_prompt,
)

nodes_with_scores = retriever.retrieve(question)
response = synthesizer.synthesize(
    query=question,
    nodes=nodes_with_scores,
)
```

직접 조립하면 다음을 제어할 수 있다.

| 단계 | 조정 가능한 것 |
|---|---|
| retrieval | top-k, similarity threshold, metadata filter, reranker |
| context | node 순서, 중복 제거, 최대 token 수, 출처 표시 |
| synthesis | prompt template, `response_mode`, 응답 형식, “근거 없으면 모름” 규칙 |
| observability | 검색 score, latency, 사용 token, source node 로그 |

Custom query engine은 보통 `query()`를 외부 API로 제공하되 내부는 `retrieve()`와 `generate/synthesize()`로 분리한다. 이 분리가 있어야 retrieval만 교체하거나, 검색 결과를 평가하고, 생성 없이 디버깅할 수 있다.

## 7. 실습 API 복원표

| 호출 | 입력 | 출력/관찰 포인트 |
|---|---|---|
| `load_data()` | `input_dir`, 파일 경로 | `list[Document]` |
| `get_nodes_from_documents(docs)` | `Document[]` | `Node[]`, chunk metadata |
| `VectorStoreIndex(nodes)` | `Node[]`, embed model | searchable index |
| `index.as_retriever(similarity_top_k=k)` | `k` | retriever 객체 |
| `retriever.retrieve(query)` | 문자열 질문 | `NodeWithScore[]` |
| `index.as_query_engine(...)` | retrieval/synthesis 설정 | query engine |
| `query_engine.query(query)` | 문자열 질문 | `Response` |
| `response.source_nodes` | 응답 객체 | 근거 Node와 score |

프레임워크 버전에 따라 import 경로와 class 이름은 바뀔 수 있으므로, 시험 답안은 decorator/import 암기보다 **입력 단위 → 검색 단위 → 생성 입력 → 근거 출력**의 계약을 설명하는 데 집중한다.

## 8. 평가 지표: 검색과 생성을 분리

### Retrieval

정답 관련 chunk 집합을 `G`, top-k 결과 집합을 `R_k`라 하면:

\[
Precision@k=\frac{|R_k\cap G|}{k},\qquad
Recall@k=\frac{|R_k\cap G|}{|G|}
\]

첫 번째 관련 결과의 순위가 `rank`일 때:

\[
RR=\frac{1}{rank},\qquad MRR=\frac{1}{Q}\sum_{j=1}^{Q}RR_j
\]

- `Precision@k`: 가져온 후보 중 관련 비율
- `Recall@k`: 필요한 근거를 얼마나 놓치지 않았는지
- `MRR`: 첫 정답을 얼마나 앞에서 찾는지
- `NDCG@k`: 관련도 등급과 순위를 함께 반영

### Generation / end-to-end

- **Faithfulness**: 답변이 검색 context에 의해 지지되는가
- **Answer relevancy**: 질문에 직접 답하는가
- **Context precision/recall**: context가 관련되고 필요한 정보를 포함하는가
- **Latency/cost**: retrieval, reranking, LLM 각각의 시간·token 비용

retriever가 좋아도 synthesis가 근거 밖 내용을 만들 수 있고, 생성 모델이 좋아도 필요한 Node를 못 찾으면 정답이 어렵다. 두 층을 따로 측정해야 병목을 찾는다.

## 9. 시험 직전 체크리스트

- [ ] `Document`와 `Node`의 차이를 말할 수 있는가?
- [ ] `chunk_size`와 `chunk_overlap`이 recall·context length에 미치는 영향을 설명할 수 있는가?
- [ ] `N_node × d` embedding matrix와 query vector shape을 쓸 수 있는가?
- [ ] `NodeWithScore.score`가 정답 확률이 아님을 아는가?
- [ ] `retriever`와 `response synthesizer`를 분리해 그릴 수 있는가?
- [ ] `source_nodes`로 citation/디버깅을 하는 이유를 말할 수 있는가?
- [ ] Precision@k, Recall@k, MRR과 Faithfulness의 평가 대상을 구분하는가?
- [ ] “LLM이 답했다”가 아니라 `query -> retrieve -> context -> synthesize` 순서를 복원할 수 있는가?

## 10. 원본 보존 정책

이 파일은 원본 notebook의 overlay 설명이다. 원본 `ipynb`와 기존 실습 가이드는 수정하지 않는다. 실행 가능한 코드는 원본 및 viewer의 practice notebook에서 확인하고, 이 문서는 객체 책임·shape·평가 포인트를 빠르게 회복하는 데 사용한다.
