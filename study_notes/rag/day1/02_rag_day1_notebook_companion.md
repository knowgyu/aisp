# RAG Day 1 Notebook Companion: LlamaIndex와 RAG App 실행 순서

대상 원본: `rag/1일차/실습 자료/Code/1. Llama_index.ipynb`, `rag/1일차/실습 자료/Code/2. RAG.ipynb`
관련 정답: `study_notes/exam_answers/rag_code_answers.md`

## 1. 객체와 데이터 흐름

```text
Document(text, metadata) -> Node(chunk) -> Index/vector store
query(str) -> Retriever(top_k Nodes) -> ResponseSynthesizer -> Response
```

| 객체/API | 역할 | 확인할 값 |
|---|---|---|
| `SimpleDirectoryReader` | 파일을 `Document`로 로드 | 문서 수, metadata/source |
| `Document` | 원문 + locator | `text`, page/file metadata |
| `SentenceSplitter` | chunk/overlap 생성 | chunk 수, 길이, 경계 |
| `VectorStoreIndex.from_documents` | embedding과 index 생성 | embedding dimension, node 수 |
| `index.as_retriever(similarity_top_k=k)` | top-k 후보 검색 | score, node text |
| `index.as_query_engine(...)` | retrieve + synthesize 묶기 | response mode, source nodes |
| `response.source_nodes` | 근거 디버깅 | source, score, chunk |

## 2. shape와 API 계약

- 한 문서의 embedding: `[D]`; N개 node matrix: `[N,D]`; query embedding: `[D]`.
- batch embedding을 쓰면 `[B,D]`; cosine similarity matrix는 `[B,N]`이다.
- `similarity_top_k=k` 결과는 node list 길이 `<= k`이며, 문서 수가 k보다 작으면 더 짧다.
- LLM 입력은 `query + retrieved context` 문자열이며 tensor shape가 아니라 token length budget이 병목이다.

## 3. 시험 실습 순서

1. 문서를 로드하고 `len(documents)`와 source metadata를 출력한다.
2. index를 만들기 전에 chunk size/overlap을 정한다.
3. query engine 답변보다 먼저 retriever node text와 score를 출력한다.
4. `k=1,3,5`로 바꿔 근거 포함 여부와 context 중복을 비교한다.
5. `response.source_nodes`로 citation locator를 확인한다.
6. prompt에 “근거에 없으면 모른다고 답한다”를 넣고 검색 실패와 생성 실패를 분리한다.

## 4. 평가와 디버깅

검색: Success@k, Recall@k, Precision@k, MRR/NDCG. 생성: exact match/F1 또는 groundedness·citation correctness. `top_k` 증가가 항상 개선은 아니다(근거 recall은 증가하지만 noise와 latency도 증가).

| 증상 | 원인 후보 | 조치 |
|---|---|---|
| 답 근거가 없음 | chunking/embedding/index | chunk와 source를 직접 출력 |
| 관련 node인데 답이 틀림 | prompt/response mode | context 순서와 prompt 점검 |
| citation 없음 | metadata 유실 | loader→node metadata 보존 |
| 느림 | 후보 수/LLM context | top-k와 chunk 길이 축소, rerank 분리 |

## 5. 코드 빈칸 복원

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

docs = SimpleDirectoryReader(input_dir="...").load_data()
nodes = SentenceSplitter(chunk_size=..., chunk_overlap=...).get_nodes_from_documents(docs)
index = VectorStoreIndex(nodes)
retriever = index.as_retriever(similarity_top_k=3)
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("...")
```

핵심은 함수명 암기보다 `Document → Node → Index → Retriever → QueryEngine → Response/source_nodes` 책임 분리다.
