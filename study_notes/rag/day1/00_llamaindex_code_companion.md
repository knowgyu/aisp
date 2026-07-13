# RAG Day 1 Companion: 검색부터 근거 생성까지

기준 자료: `study_notes/rag/day1/01_rag_day1_lecture_review.md`, LlamaIndex/RAG 원본 노트북. 목적은 API 이름 나열이 아니라 각 객체의 책임과 shape를 시험 답안으로 복원하는 것이다.

## 1. 데이터 흐름

```text
Document(text, metadata)
 → Node/chunk(text, metadata)
 → embedding [D]
 → VectorStoreIndex
 → Retriever(top_k Nodes)
 → prompt(context + question)
 → LLM/ResponseSynthesizer
```

`Document`는 원문과 출처를 보존하고, `Node`는 검색 단위다. `VectorStoreIndex`는 node embedding을 저장하며 `as_retriever(similarity_top_k=k)`가 후보를 반환한다. Query engine은 retriever와 response synthesizer를 묶는다.

## 2. 핵심 객체와 shape

| 객체/호출 | 역할 | 결과 |
|---|---|---|
| `SimpleDirectoryReader` | 파일을 `Document`로 로드 | 문서 목록, metadata 포함 |
| `SentenceSplitter` | 문서를 chunk로 분할 | 각 node의 text, `chunk_size`, overlap |
| embedding model | text를 벡터로 변환 | query/document 각각 `[D]` 또는 batch `[N,D]` |
| `VectorStoreIndex.from_documents` | index 구축 | node→vector 저장 |
| `retriever.retrieve(query)` | top-k 후보 검색 | `NodeWithScore[]`, score scalar |
| `query_engine.query(q)` | 검색·프롬프트·생성 오케스트레이션 | response + source nodes |

## 3. 검색과 생성 평가를 분리하기

검색 평가에서 `Success@k`는 정답 근거가 top-k에 포함됐는지, `MRR`은 첫 정답 순위의 역수다. `Precision@k`는 반환 후보 중 관련 비율, `Recall@k`는 관련 문서 중 찾은 비율이다. 생성 답변은 exact match만으로 충분하지 않으므로 faithfulness(근거와 일치), answer relevance, citation coverage를 별도로 본다.

## 4. 시험형 코드 복원

```python
nodes = splitter.get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes, embed_model=embed_model)
retriever = index.as_retriever(similarity_top_k=top_k)
results = retriever.retrieve(question)
context = "\n\n".join(r.node.get_content() for r in results)
```

query embedding과 node embedding의 차원 `D`가 같아야 cosine/dot similarity가 정의된다. reranker를 쓴다면 retriever의 넓은 후보를 cross-encoder가 재정렬한다. prompt에는 context뿐 아니라 source metadata를 함께 넣어 근거 추적을 가능하게 한다.

## 5. 실패 진단

- 정답이 검색되지 않음: chunk 크기/overlap, embedding model, top-k, query rewrite를 점검한다.
- 검색은 맞지만 답변이 틀림: context serialization, prompt, citation 요구, generator를 점검한다.
- 긴 문서에서 느림: ANN index(HNSW 등), embedding cache, rerank 후보 수를 점검한다.
- 최신 정보가 필요함: 정적 index와 web/database retriever의 freshness 경계를 명시한다.

원본을 수정하지 않고 viewer용 해설만 추가했다.
