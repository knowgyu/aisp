# RAG Day 2 Practice 01. MCP 기반 평가와 업그레이드 코드 학습 가이드

오른쪽 실습본은 `4_RAG_framework_evaluation_with_MCP.ipynb`의 핵심 코드 다섯 셀을 비워 둔 시험 대비 버전이다.

원본: `rag/2일차/실습 자료/Code/4_RAG_framework_evaluation_with_MCP.ipynb`

## 전체 흐름

```text
CRAG sample -> LLM judge -> parse_response -> CRAG_evaluation
            -> 기존 KG error 분석
            -> MCP client/tool discovery
            -> FunctionAgent 기반 KGQueryEngineWithMCP
            -> RAGwithMCP
```

## Drill 지도

| Drill | 원본 셀 | 복원할 핵심 |
|---:|---:|---|
| 1 | 017 | LLM judge JSON 응답을 score로 parsing |
| 2 | 019 | question/ground truth/prediction 평가 함수 |
| 3 | 022 | Retriever + KGQueryEngine + Reader를 묶는 RAG |
| 4 | 046 | MCP tool을 FunctionAgent에 연결하고 event stream 처리 |
| 5 | 055 | MCP retrieval과 Reader generation을 묶는 async RAG |

## 핵심 자료구조

- evaluation input: `question`, `ground_truth`, `prediction`
- evaluation output: `-1 | 0 | 1` 또는 exact/correct/miss 분류
- MCP tools: `list[BaseTool]`
- agent result: async handler와 `ToolCall`, `ToolCallResult` event
- final RAG output: `{retrieved_results, answer}`

## 실수 포인트

- MCP server URL/SSE transport는 실행 환경에 의존한다.
- query time을 prompt에 넣지 않으면 동적 사실 질문이 틀릴 수 있다.
- tool output이 길면 context limit을 넘을 수 있다.
- evaluation LLM의 JSON이 깨질 수 있으므로 parser 경계가 필요하다.
- API key는 notebook에 직접 저장하지 않는다.

정답은 `RAG 코드 실습 정답·해설지`의 같은 제목 절에서 확인한다.
