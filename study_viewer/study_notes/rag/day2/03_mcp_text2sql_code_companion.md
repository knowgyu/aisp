# RAG Day 2. MCP Text2SQL 코드 동반 복습

대상 원본은 `rag/2일차/mcp-text2sql/README.md`, `server.py`, `sample_db.py`, `rag/2일차/NL2SQL___mcp.pdf` 및 `study_notes/rag/day2/01_mcp_text2sql_review.md`다. MCP server가 LLM을 대신하는 것이 아니라, Host가 선택한 resource/tool을 안전하게 실행한다는 경계를 중심으로 복습한다.

## 호출 흐름

```text
사용자 질문 → MCP Host/LLM
  → Resource: sqlite://schema / get_schema
  → Text2SQL 생성
  → Tool: query_db(sql)
  → read-only SQLite 결과(list[dict])
  → LLM이 결과를 자연어로 설명
```

| 구성요소 | 책임 | 시험 함정 |
|---|---|---|
| Host | UI, 대화, LLM 호출, tool 선택 | server가 질문 의도를 해석한다고 생각하지 않기 |
| Client | server 연결과 호출 전달 | DB 권한을 소유하지 않음 |
| Server | resource/tool 제공, 검증, DB 실행 | 무제한 SQL 실행기가 아님 |
| SQLite | schema/row 저장과 SQL 실행 | LLM에 직접 연결하지 않기 |

## Resource·Tool·Prompt

- **Resource**: `get_schema` 또는 `sqlite://schema`처럼 모델에 table/column 정보를 제공하는 읽기 context.
- **Tool**: `list_tables`, `sample_data`, `query_db`처럼 실행되는 함수. 반환 구조는 보통 JSON 직렬화 가능한 `list[dict]`다.
- **Prompt**: SQL 생성 규칙을 재사용하는 템플릿. 실행 자체가 아니다.

## 데이터 구조와 안전 경계

| 입력/출력 | 권장 형태 |
|---|---|
| schema | table별 column/type 문자열 또는 구조화 JSON |
| SQL input | 단일 statement 문자열 |
| query result | `list[dict[str, Any]]` |
| 오류 | 사용자에게 내부 경로/비밀정보를 노출하지 않는 명시적 error |

`SELECT`만 허용한다고 문자열 prefix만 검사하면 주석, 다중 statement, 우회 문법을 놓칠 수 있다. read-only connection/transaction, SQL parser/AST statement 검사, 허용 table/column 정책, 결과 row 제한과 timeout을 함께 둔다. `DROP`, `DELETE`, `UPDATE`, `INSERT`, 임의 파일 접근은 거부한다.

## API 복원 순서

1. `list_tables`로 존재하는 table을 확인한다.
2. `get_schema`로 정확한 column 이름/type을 grounding한다.
3. LLM이 schema에 맞는 `SELECT`를 만든다.
4. `query_db(sql)` 호출 전에 read-only 검증을 통과시킨다.
5. 결과 row와 오류를 Host에 반환한다.
6. Host/LLM이 결과를 답변으로 렌더링한다.

예시 질문 “부서별 평균 급여”라면 `GROUP BY`와 aggregate 결과를 확인하되, 없는 column을 추측하지 않는다. 결과가 비어 있으면 “0”으로 보정하지 말고 empty result로 처리한다.

## 시험 체크리스트

- MCP는 모델이 아니라 **Host–Client–Server 프로토콜**이다.
- schema resource는 SQL 생성 정확도를 높이고 tool은 실행 권한을 가진다.
- Text2SQL 생성과 SQL 실행을 분리해야 감사·권한·재현성이 좋아진다.
- metric은 SQL 실행 성공률/정확도와 답변 groundedness를 분리해 기록한다.
- 원본의 `cwd`와 `claude_desktop_config.json` 경로는 환경별로 바뀌므로 문서 경로를 그대로 실행하지 않는다.
