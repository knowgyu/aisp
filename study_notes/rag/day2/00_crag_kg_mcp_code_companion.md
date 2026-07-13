# RAG Day 2 MCP Companion: Text2SQL 안전 경계

기준 자료: `study_notes/rag/day2/01_mcp_text2sql_review.md`와 MCP/Text2SQL 실습. MCP는 LLM 자체가 아니라 Host–Client–Server 사이의 **resource, tool, prompt 계약**이다.

## 1. 호출 흐름

```text
User question
 → Host가 LLM에 질문 + schema(Resource) 제공
 → LLM이 SQL 문자열 생성
 → Client가 Server Tool 호출
 → Server가 read-only 검증 후 DB 실행
 → rows(JSON) 반환
 → Host/LLM이 답변과 근거 생성
```

| 구성요소 | 코드 책임 | 하지 않는 일 |
|---|---|---|
| Host | 대화, LLM 호출, 승인, 결과 표시 | DB를 직접 무제한 노출 |
| Client | server 연결, resource/tool 호출 전달 | SQL 정책을 우회 |
| Server | schema resource, SQL tool, 입력 검증 | 사용자 질문을 독자적으로 이해 |
| Database | schema와 row, SQL 실행 | LLM prompt를 실행 |

## 2. API와 데이터 shape

```python
@mcp.resource("sqlite://schema")
def schema() -> str: ...

@mcp.tool()
def run_read_only_sql(sql: str) -> list[dict]: ...

@mcp.prompt()
def text2sql(question: str) -> str: ...
```

Resource는 `str` context, tool 입력은 SQL `str`, 반환은 행 목록 `list[dict]` 또는 JSON이다. schema는 table/column/타입을 제공하고, row 결과는 Host가 표나 요약으로 렌더링한다. Prompt는 실행기가 아니라 일관된 SQL 생성 규칙을 제공한다.

## 3. 안전·평가 체크리스트

1. DB 연결을 read-only 또는 별도 transaction으로 제한한다.
2. `SELECT` prefix만 보지 말고 SQL parser/AST로 단일 statement와 허용 테이블·함수를 검증한다.
3. timeout, row limit, `LIMIT` 강제, parameter binding, 민감 column 차단을 둔다.
4. 실행 전 SQL과 영향 범위를 로그로 남기되 비밀값은 마스킹한다.
5. gold 질문–SQL 쌍으로 execution accuracy와 result correctness를 평가한다.
6. 생성 답변은 schema grounding, 실행 성공률, latency, refusal/safety violation을 함께 측정한다.

## 4. 흔한 오답

- “MCP Server가 LLM을 호출해 Text2SQL을 수행한다”: 일반적인 구조에서 LLM은 Host 쪽이다.
- “Resource와 Tool은 같다”: Resource는 읽는 context, Tool은 실행 가능한 동작이다.
- “SQL이 실행되면 정답”: 잘못된 join·필터는 실행되지만 의미적으로 오답이다.
- GraphRAG는 MCP의 하위 기능이 아니다. MCP는 연결 계약이고 GraphRAG는 검색/지식 표현 전략이다.

## 5. CRAG·Web·KG를 함께 읽기

CRAG는 검색 결과를 평가한 뒤 부족하면 Web retriever 또는 Knowledge Graph retriever로 보완한다. Web 경로는 HTML 파싱→chunk→embedding/retrieval→Reader이고, KG 경로는 entity/relation 추출→triplet/subgraph→graph-aware prompt/Reader다. MCP는 이 retriever나 평가 도구를 Host가 안전하게 호출하기 위한 transport/contract 층으로 구분한다.

- Web retriever 결과: text chunk와 URL metadata, 보통 `[k]`개 후보.
- KG retriever 결과: entity/node/edge/path/subgraph, 관계 근거와 multi-hop 정보.
- Reader/LLM 입력: 질문 + 선택된 근거를 직렬화한 prompt; 출력은 answer와 source/citation.
- 평가: retrieval recall/precision과 answer faithfulness를 분리하고, MCP tool latency·실패율도 기록한다.
