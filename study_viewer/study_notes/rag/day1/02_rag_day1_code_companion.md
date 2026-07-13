# RAG Day 1. LlamaIndex 실습 코드 동반 복습

대상 원본은 `rag/1일차/실습 자료/Code/1. Llama_index.ipynb`, `2. RAG.ipynb`와 `study_notes/rag/day1/01_rag_day1_lecture_review.md`다. 이 문서는 원본을 보존한 채 코드 객체와 데이터 흐름을 시험 관점에서 정리한다.

## 전체 데이터 흐름

```text
Text/PDF → Document → SentenceSplitter → Node(chunk) → VectorStoreIndex
→ Retriever(top-k Node) → ResponseSynthesizer/LLM → Response
```

| 객체/API | 역할 | 확인할 값 |
|---|---|---|
| `SimpleDirectoryReader` | 파일을 `Document`로 로드 | `documents[i].text`, metadata |
| `Document` | 원문과 `doc_id`를 담는 단위 | `Document(text=..., id_=...)` |
| `SentenceSplitter` | 문서를 겹치는 chunk로 분할 | `chunk_size`, `chunk_overlap` |
| `VectorStoreIndex.from_documents` | embedding과 index 생성 | node 수, index 구조 |
| `index.as_retriever()` | 질문을 관련 Node로 검색 | `retrieve(query)` 결과 |
| `index.as_query_engine()` | 검색+응답 합성 facade | `query(question)` |
| `get_response_synthesizer` | retrieved context를 LLM 입력으로 조합 | `response_mode` |

## shape와 데이터 구조

LlamaIndex는 PyTorch tensor가 아니라 객체/문자열 흐름이 중심이다.

| 단계 | 대표 구조 |
|---|---|
| 문서 | `list[Document]` |
| chunk | `list[TextNode]`; 각 `text`는 문자열, `metadata`는 dict |
| embedding | 각 Node당 고정 길이 `list[float]` 벡터(모델 차원에 따름) |
| retrieval | `list[NodeWithScore]`; `node`, `score` 포함 |
| answer | `Response`; `str(response)`로 출력 |

`chunk_size=200, chunk_overlap=50`은 200 토큰 안팎 chunk를 만들되 인접 chunk에 50 토큰 문맥을 겹친다. 문자 수와 토큰 수를 혼동하지 말고 `tiktoken.encoding_for_model("gpt-3.5-turbo")`로 실제 token 수를 점검한다.

## Query Engine 내부 복원

```python
nodes = retriever.retrieve(query_str)
response_obj = response_synthesizer.synthesize(query_str, nodes)
```

`as_query_engine()`은 retriever와 response synthesizer를 묶은 편의 API다. 직접 `CustomQueryEngine`을 만들 때도 **검색 결과가 먼저**이고, LLM은 retrieved context를 prompt에 넣어 답을 합성한다. `PromptTemplate`의 `{context_str}`, `{query_str}`를 빠뜨리면 질문과 근거가 전달되지 않는다.

## CRUD API와 index 수명주기

- `index.insert(Document(...))`: 새 문서 추가
- `index.update_ref_doc(doc, update_kwargs=...)`: 같은 `doc_id`의 내용 갱신
- `index.refresh_ref_docs([doc])`: 여러 참조 문서를 새 내용으로 동기화
- `index.delete_ref_doc(doc_id, delete_from_docstore=True)`: 문서 삭제
- `index.ref_doc_info.keys()`: 참조 문서 ID 점검

추가/수정 직후 같은 질문을 다시 보내어 이전 답변이 남아 있지 않은지 확인한다.

## 평가와 함정

| 평가 질문 | 확인 포인트 |
|---|---|
| 데이터에 직접 답이 있는가? | retrieved passage에 근거 문장이 존재하는가 |
| 데이터와 무관한 질문인가? | 검색 context 밖의 LLM 사전지식으로 답하지 않는가 |
| 간접 추론 질문인가? | 답과 근거를 함께 요구하고 retrieved Node를 출력하는가 |
| chunk 경계가 문제인가? | `chunk_overlap`과 top-k를 조정해 재현하는가 |

정답이 맞아도 근거가 검색되지 않았다면 RAG 품질이 검증된 것이 아니다. `ret_passages[i].text`와 score를 먼저 출력한다. API key는 notebook에 직접 기록하지 말고 환경변수로 주입한다.

## 시험 직전 체크

- `Document → Node → Retriever → Synthesizer` 순서를 말할 수 있는가?
- `top-k`는 생성 길이가 아니라 검색 후보 수라는 점을 구분하는가?
- `chunk_size`와 `chunk_overlap`이 recall/context 비용에 미치는 영향을 설명하는가?
- 검색 실패와 LLM hallucination을 별도 원인으로 진단하는가?
