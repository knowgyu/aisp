# RAG Day 2 MCP Notebook Companion: 평가와 외부 도구 경계

대상 원본: `rag/2일차/실습 자료/Code/4_RAG_framework_evaluation_with_MCP.ipynb`, `1. Data_preprocessing.ipynb`, `2. Task_1.ipynb`, `3. Task_2.ipynb`
관련 해설: `study_notes/rag/day2/01_mcp_text2sql_review.md`, `02_graphrag_crag_review.md`

## 1. MCP 구성

```text
Host/LLM -> MCP Client -> MCP Server -> resource/tool -> DB, web, KG
```

| API/객체 | 역할 | 데이터 계약 |
|---|---|---|
| `FastMCP(name)` | server registry | server name |
| `@mcp.resource(uri)` | 읽기 context/schema | URI → text/JSON |
| `@mcp.tool()` | 실행 동작 | validated args → result |
| MCP Client session | list/read/call | resource/tool discovery |
| evaluator tool | retrieval/generation 평가 | score + evidence |

Text2SQL에서 LLM이 SQL을 생성하는 위치는 Host이며, Server는 schema resource와 read-only SQL tool을 제공한다. SQL parser/AST, allowlist, read-only connection, timeout/row limit을 함께 둔다.

## 2. MCP 평가 notebook 흐름

```text
question + reference -> retrieve/tool call -> context/result
-> answer generation -> retrieval metrics + answer metrics
```

- retrieval 결과: candidate list `top_k` (각 node에는 text, score, source metadata).
- embedding matrix: `[N,D]`, query: `[D]`, similarity: `[N]` 또는 batch `[B,N]`.
- 평가 batch: questions `[B]`, retrieved contexts `[B,K]`(문자열/노드), answers `[B]`.

검색 지표는 `Recall@K`, `Precision@K`, `MRR`, `NDCG@K`; 생성 지표는 exact match/F1 또는 faithfulness, answer relevancy, citation correctness로 분리한다. latency와 token cost도 함께 기록한다.

## 3. CRAG/Web/GraphRAG 흐름

| notebook | 핵심 객체 흐름 | 실수 포인트 |
|---|---|---|
| Data preprocessing | raw text → clean/chunk → metadata | source locator와 label 보존 |
| Web RAG | query → web retriever → reader/answer | 검색 결과 HTML 전체를 그대로 넣지 않기 |
| KG RAG | entities/relations → graph → Cypher/SPARQL retrieval | entity ID와 relation 방향 확인 |

GraphRAG는 vector similarity만이 아니라 entity-hop/관계 경로로 근거를 확장한다. context는 `[문서 chunk, graph path, source]`로 묶고, graph hop이 증가할수록 noise와 latency를 점검한다.

## 4. 실행 전 체크리스트

1. tool/resource 목록을 먼저 확인한다.
2. tool 입력 schema와 반환 JSON shape를 출력한다.
3. 실패/빈 결과를 정상적인 구조화 오류로 처리한다.
4. 생성 답변과 tool evidence를 함께 저장한다.
5. 같은 질문에 top-k, reranker, graph hop을 바꿔 metric과 latency를 비교한다.

## 5. 빈칸 답안

```python
@mcp.resource("sqlite://schema")
def schema() -> str: ...

@mcp.tool()
def run_read_only_sql(sql: str) -> list[dict]: ...
```

MCP는 RAG 자체가 아니라 외부 context/tool을 표준화하는 연결 계층이다. Resource는 context 제공, Tool은 효과가 있는 실행, Prompt는 재사용 지시 템플릿이다.
