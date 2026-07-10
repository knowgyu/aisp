# RAG Day 2 Practice 04. Knowledge Graph RAG 코드 학습 가이드

원본: `rag/2일차/실습 자료/Code/3. Task_2.ipynb`

목표: 자연어 질문에서 entity/API argument를 만들고, decision tree로 Mock KG를 호출한 뒤 Web retrieval과 합치는 흐름을 복원한다.

## Drill 지도

| Drill | 원본 셀 | 핵심 |
|---:|---:|---|
| 1 | 013 | entity extraction prompt message 구성 |
| 2 | 015 | LLM query generation과 JSON parsing |
| 3 | 021 | finance domain decision tree/API 실행 |
| 4 | 025 | `KGQueryEngine` 추상화 |
| 5 | 031 | KG-only RAG |
| 6 | 036 | Web search + KG hybrid RAG |

## Query generation contract

```text
natural-language query
  -> domain / market identifier / ticker / metric / datetime
  -> validated structured JSON
  -> API decision tree
  -> KG results
```

LLM output을 바로 함수 인자로 쓰지 말고 schema/type/value를 검증해야 한다.

## Hybrid context

```text
Web chunks: 넓은 coverage, 비정형
KG results: 정밀한 structured fact
        ↓ merge / dedupe / time alignment
Reader prompt
```

정답은 `RAG 코드 실습 정답·해설지`에서 확인한다.
