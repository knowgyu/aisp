# RAG Day 2 Practice 02. CRAG 데이터 전처리 코드 학습 가이드

원본: `rag/2일차/실습 자료/Code/1. Data_preprocessing.ipynb`

목표: CRAG JSONL/BZ2 dataset을 읽고 domain/question type/dynamism을 분석한 뒤, Web search HTML을 RAG가 검색할 text chunk로 바꾼다.

## Drill 지도

| Drill | 원본 셀 | 핵심 |
|---:|---:|---|
| 1 | 016 | BZ2 JSONL을 `list[dict]` dataset으로 읽기 |
| 2 | 019 | baseline LLM answer 생성 |
| 3 | 021 | domain별 대표 example grouping |
| 4 | 044 | HTML parse → sentence offsets → chunk 추출 |

## 데이터 흐름

```text
.jsonl.bz2
  -> each line: JSON object
  -> query / answer / domain / question_type / static_or_dynamic
  -> search_results[page_result HTML]
  -> BeautifulSoup text
  -> sentence offsets
  -> chunks
```

## 확인 포인트

- JSONL은 전체가 하나의 JSON array가 아니라 한 줄에 한 object다.
- `bz2.open(..., "rt")`로 text mode를 사용한다.
- HTML tag를 제거하되 문장 사이 공백을 보존한다.
- dataset 통계는 모델 성능을 domain/type/dynamism별로 해석하는 기준이다.

정답은 `RAG 코드 실습 정답·해설지`에서 확인한다.
