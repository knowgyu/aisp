# RAG Day 2 MCP 시험 동반 노트: Text2SQL·GraphRAG·평가

대상 원본: `rag/2일차/NL2SQL___mcp.pdf`, `Graph_RAG.pdf`, `rag/2일차/실습 자료/Code/4_RAG_framework_evaluation_with_MCP.ipynb` 및 `mcp-text2sql/`.

## 1. MCP를 포함한 흐름

```text
question + query_time
 -> LLM agent
 -> MCP Client -> MCP Server(tools/resources/prompts)
 -> SQL/KG/web 결과
 -> reader + citations -> answer
```

- **Server**는 DB/knowledge graph/API를 안전한 도구로 노출한다.
- **Client**는 LLM runtime과 server를 연결한다. `BasicMCPClient(url)`과 `McpToolSpec(client=...)`로 tool 목록을 얻는다.
- **Agent**는 `FunctionAgent`가 schema를 보고 적절한 tool을 호출한다.
- MCP는 모델 자체가 아니라 **도구와 context를 표준 인터페이스로 연결하는 protocol**이다.

## 2. 핵심 API와 객체

```python
mcp_client = BasicMCPClient(external_mcp_server)
mcp_tool = McpToolSpec(client=mcp_client)
tools = await mcp_tool.to_tool_list_async()
engine = KGQueryEngineWithMCP(mcp_tool, model="gpt-4o-mini")
rag = RAGwithMCP(mcp_tool)
result = await rag.inference(query, search_results=[], query_time=query_time, topk=5)
```

`RAGwithMCP`는 대체로 `retrieve()`에서 KG/MCP 결과를 모으고 `Reader`가 query, query_time, references를 사용해 답한다. 결과 dict에서 `retrieved_results`와 `answer`를 분리해 확인한다.

## 3. CRAG 평가

CRAG의 자동/LLM judge 결과는 예측을 다음처럼 분류한다.

| 분류 | 의미 |
|---|---|
| `correct` | 정답과 의미상 일치 |
| `incorrect` | 근거가 있어도 답이 틀림 |
| `missing` | 검색/응답에 필요한 정보가 없음 |
| `exact` | predefined answer와 정확히 일치하는 강한 기준 |

실습의 `generate_answer()`는 judge prompt를 호출하고 `parse_response()`가 JSON의 `accuracy` 등을 읽는다. 따라서 parser는 malformed JSON, boolean/string 혼용, 누락 key를 방어해야 한다. 주요 aggregate는 accuracy, exact accuracy, miss 비율이다.

## 4. MCP 설계·실패 포인트

- query time을 prompt에 명시한다. “현재” 같은 상대 표현은 질문이 입력된 시각에 anchor해야 한다.
- KG 검색 결과가 지나치게 길면 hallucination과 context limit가 증가한다. top-k, structured result, `Rerank`/요약 단계로 줄인다.
- tool schema와 server URL은 환경에 따라 달라질 수 있으므로 notebook의 placeholder URL을 그대로 배포하지 않는다.
- MCP agent가 틀린 답을 내면 (1) query 생성, (2) tool 반환, (3) reader prompt, (4) 시간 조건을 순서대로 출력해 원인을 분리한다.
- tool 권한, SQL injection, tenant 분리, timeout도 production 경계다.

## 5. Day 1과 비교

| 항목 | Day 1 | Day 2 MCP |
|---|---|---|
| 외부 지식 | 정적 node/vector index | KG/DB/API tool |
| 호출 방식 | retriever가 직접 검색 | agent가 tool을 선택·호출 |
| 평가 | retrieval + groundedness | CRAG correct/missing/exact + latency/cost |
| 위험 | chunk/embedding 오류 | tool 선택, schema, 시간, context 폭증 |
