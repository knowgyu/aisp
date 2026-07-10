# RAG Day 2 Practice 03. Web Retriever와 Reader 코드 학습 가이드

원본: `rag/2일차/실습 자료/Code/2. Task_1.ipynb`

목표: framework 없이 구현한 retriever와 LlamaIndex retriever를 비교하고, prompt/Reader/RAG 조합을 직접 복원한다.

## Drill 지도

| Drill | 원본 셀 | 핵심 |
|---:|---:|---|
| 1 | 007 | HTML parse와 chunk extractor |
| 2 | 012 | embedding 기반 `BaseRetriever` |
| 3 | 016 | LlamaIndex `Document`, splitter, vector index retriever |
| 4 | 023 | query와 top-k chunk를 prompt message로 구성 |
| 5 | 025 | Reader의 LLM 호출 |
| 6 | 029 | Retriever와 Reader를 묶는 RAG class |

## Retriever 핵심 수식

query embedding $q$와 chunk embedding $d_i$의 cosine similarity:

$$
\operatorname{sim}(q,d_i)=\frac{q\cdot d_i}{\|q\|\|d_i\|}
$$

상위 `topk` chunk를 Reader reference로 전달한다.

## 디버깅 순서

1. HTML parse 결과가 비어 있지 않은가?
2. chunk 길이와 개수가 과도하지 않은가?
3. retrieved chunk에 정답 근거가 있는가?
4. 근거가 있는데 답이 틀리면 prompt/Reader를 본다.
5. 근거가 없으면 embedding/chunk/top-k를 본다.

정답은 `RAG 코드 실습 정답·해설지`에서 확인한다.
