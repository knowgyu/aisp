# RAG Day 1 시험 동반 노트: 검색부터 답변까지

대상 원본: `rag/1일차/`의 IR/RAG PDF·HTML, `rag/1일차/실습 자료/Code/1. Llama_index.ipynb`, `2. RAG.ipynb`.

## 1. 전체 data flow

```text
문서 -> parser/metadata -> chunk(nodes) -> index
질문 -> retriever(top-k nodes) -> reranker/response synthesizer -> LLM 답변
```

| 객체 | 역할 | 자주 보는 값/shape |
|---|---|---|
| `Document`/`Node` | 원문과 metadata 보관 | text 문자열, source/id metadata |
| `VectorStoreIndex` | node embedding과 검색 구조 | N개 node, embedding `[N, D]` |
| `retriever` | 질문과 가까운 node 반환 | `list[NodeWithScore]`, 길이 `k` |
| `query_engine` | retrieve+generate 조립 | query 문자열→response |
| response synthesizer | context를 answer/summary로 결합 | answer 문자열, source nodes |

## 2. LlamaIndex 실습 API

```python
nodes = SentenceSplitter(chunk_size=..., chunk_overlap=...).get_nodes_from_documents(docs)
index = VectorStoreIndex(nodes)
retriever = index.as_retriever(similarity_top_k=2)
query_engine = index.as_query_engine(similarity_top_k=2, response_mode="compact")
answer = query_engine.query(question)
retrieved = retriever.retrieve(question)
```

- `chunk_size`가 너무 작으면 답에 필요한 문장이 잘리고, 너무 크면 noise와 token 비용이 늘어난다.
- `chunk_overlap`은 경계에서 문맥이 끊기는 것을 줄이지만 index 크기를 키운다.
- `response_mode="compact"`는 여러 context를 압축해 전달하는 전략이지 검색 품질 자체를 보장하지 않는다.
- `query_engine` 내부 결과와 `retriever.retrieve()` 결과를 따로 출력하면 검색 실패와 생성 실패를 분리할 수 있다.

## 3. Sparse/Dense/Rerank

| 단계 | 대표 기법 | 강점/약점 |
|---|---|---|
| sparse | inverted index, TF-IDF, BM25 | 고유명사·숫자·정확한 단어에 강함 |
| dense | embedding + cosine/dot product | 표현이 다른 paraphrase에 강함 |
| rerank | cross-encoder | query-document 상호작용이 강하지만 후보마다 계산 |

Cosine similarity는 `q·d/(||q||||d||)`이며, vector search의 top-k 후보를 의미 유사도 순으로 정렬한다. 실제 pipeline은 BM25+dense 후보를 합친 뒤 cross-encoder를 적용할 수 있다.

## 4. 평가 지표

- `Success@k`: top-k 안에 관련 문서가 하나라도 있는가.
- `Precision@k`: top-k 중 관련 문서 비율.
- `Recall@k`: 전체 관련 문서 중 top-k에 포함된 비율.
- `MRR`: 첫 관련 문서 순위의 역수 평균.
- `NDCG@k`: 관련도 등급과 순위를 함께 반영.
- generation은 `groundedness`, `answer relevance`, `context relevance`를 별도로 본다.

**핵심 답안:** 검색 결과에 정답 근거가 없으면 LLM을 바꿔도 해결되지 않는다. 먼저 retrieval recall/순위를 확인한 뒤 prompt, synthesizer, generator를 조정한다.

## 5. 빈칸/디버깅 포인트

1. `fit`/indexing은 문서 corpus에 한 번, query는 동일 embedding model로 encode한다.
2. `similarity_top_k=1`은 비교 실험에는 유용하지만 다단계 답변에는 recall이 부족할 수 있다.
3. source metadata를 node에 유지해야 citation과 error analysis가 가능하다.
4. 답변이 맞더라도 retrieved node가 무관하면 parametric memory일 수 있으므로 성공으로 단정하지 않는다.
