# RAG Day 2. CRAG · Knowledge Graph · MCP 코드 동반 복습

대상 원본은 `rag/2일차/실습 자료/Code/1. Data_preprocessing.ipynb`부터 `4_RAG_framework_evaluation_with_MCP.ipynb`까지다. Day 2의 핵심은 LlamaIndex 문법을 더 외우는 것이 아니라, **질문 종류에 따라 Web 검색·Knowledge Graph·MCP Tool 호출을 어떤 순서로 조합하는지**다.

## 1. 전체 파이프라인

```text
CRAG JSONL 한 행
  ├─ query / answer / query_time
  ├─ search_results(HTML)
  └─ domain / question_type / static_or_dynamic
        ↓
HTML 정리 → chunk → Web Retriever
질문 분석 → KG API 또는 MCP Tool
        ↓
Web context + KG/MCP result
        ↓
Reader(OpenAI LLM) → 짧은 최종 답변
```

`Reader`는 그림에서 Generator 역할이다. 검색하지 않고, 이미 전달받은 references를 프롬프트에 넣어 답변을 만든다.

## 2. CRAG 데이터와 Web RAG

| 필드/객체 | 역할 | 자료형·형태 |
|---|---|---|
| `dataset` | JSONL을 읽은 샘플 목록 | `list[dict]` |
| `item['query']` | 사용자 질문 | `str` |
| `item['search_results']` | 웹 검색 페이지 목록 | `list[dict]` |
| `parse_htmls()` | `page_result` HTML에서 텍스트 추출 | `list[str]` documents |
| chunk extractor | 문서를 문장/길이 기준 chunk로 분리 | `list[str]` chunks |
| Retriever | query와 chunk의 embedding 유사도 검색 | top-k chunk `list[str]` |
| Reader | references를 보고 답변 생성 | `str` |

`search_results`는 이미 외부 검색으로 얻은 후보 문서다. 따라서 이 실습의 Retriever는 인터넷을 다시 검색하는 것이 아니라, **각 질문에 딸려 온 HTML 후보를 정리·벡터화·재정렬**한다.

## 3. Task 1의 수동 RAG와 LlamaIndex RAG

수동 `BaseRetriever`는 query와 candidate chunk를 embedding API에 넣고 score를 계산해 top-k를 고른다. LlamaIndex 버전은 `Document → VectorStoreIndex → as_retriever()`로 같은 책임을 짧게 감싼다.

```text
query str → query embedding [D]
chunk str → chunk embedding [D]
score(query, chunk) → scalar
Top-k → Reader input list
```

embedding의 `D`는 같은 모델을 썼을 때만 비교 가능하다. `topk=5`는 embedding 차원이 아니라 **Reader에 넘길 후보 개수**다.

## 4. Knowledge Graph 경로

금융처럼 날짜·회사·지표가 구조화된 질문은 Web snippet만으로 부족할 수 있다.

```text
자연어 query
 → LLM query generator
 → 구조화된 entity / metric / datetime
 → KG API 호출
 → kg_results
 → Reader
```

기존 `KGQueryEngine`은 LLM이 구조화 query를 만들고 Python decision tree가 어떤 REST API를 부를지 결정한다. 문제는 LLM 출력이 매번 조금 달라질 수 있고, KG 결과가 길어 context를 오염시킬 수 있다는 점이다.

## 5. MCP가 교체하는 부분

MCP는 RAG 전체를 대체하지 않는다. **KG 접근 방식**을 decision tree + 직접 REST 호출에서 tool calling으로 바꾼다.

```python
mcp_client = BasicMCPClient(external_mcp_server)
mcp_tool = McpToolSpec(client=mcp_client)
tools = await mcp_tool.to_tool_list_async()
agent = FunctionAgent(tools=tools, llm=llm, system_prompt=SYSTEM_PROMPT)
```

| 객체 | 책임 |
|---|---|
| `BasicMCPClient` | SSE/HTTP 기반 MCP server 연결 |
| `McpToolSpec` | server의 tool schema를 LlamaIndex Tool로 변환 |
| `FunctionAgent` | 질문을 보고 tool 선택·인자 생성·호출·후속 답변 생성 |
| `Context` | agent workflow 상태 보관 |
| `ToolCallResult.tool_output` | KG Tool의 raw 결과 |
| `await handler` / `str(response)` | tool 결과를 반영한 최종 자연어 답변 |

`agent.run(question, ctx=...)`은 즉시 최종 문자열을 주는 일반 함수가 아니라 비동기 workflow handler를 반환한다. `stream_events()`에서 `ToolCall`과 `ToolCallResult`를 보면 실제 도구 호출을 확인할 수 있고, 마지막 `response`는 raw KG JSON이 아니라 Agent가 정리한 답변이다.

## 6. `query_time`과 context limit

CRAG 동적 질문은 “현재/최근/올해”의 기준일이 중요하다.

```text
Query: ...
Query time: 2024-...
```

처럼 명시하지 않으면 Agent가 시간 범위를 추측한다. 또 Tool output이 너무 길면 context window를 초과하거나 핵심을 놓친다. 그래서 Day 2는 `query_time`을 질문에 포함하고, 필요하면 더 큰 context window 모델을 사용하며, 검색 결과 길이/top-k를 제한한다.

## 7. 시험형 복원 순서

1. `query`, `search_results`, `query_time` 중 무엇이 input인지 적는다.
2. Web chunk와 KG result를 각각 어디서 만드는지 구분한다.
3. Retriever는 context를 찾고 Reader는 답변을 만든다고 분리해 설명한다.
4. MCP에서는 tool 목록 → Agent → tool output → 최종 response 순서로 적는다.
5. 시간 누락, 너무 긴 context, 잘못된 entity/metric JSON을 실패 원인으로 든다.
