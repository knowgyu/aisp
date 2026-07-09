# AISP study viewer / 교안 제작 작업 규칙

이 문서는 `aisp` 저장소에서 새 강의자료가 들어왔을 때, 기존 방식과 같은 형태로 웹 학습 뷰어와 Markdown 교안을 확장하기 위한 작업 지침이다.

## 1. 기본 원칙

- 원본 PDF, HTML, ipynb는 직접 수정하지 않는다.
- `study_notes/` 아래에 사람이 읽는 복습 교안 Markdown을 새로 만든다.
- 웹 배포물은 `study_viewer/` 아래에 복사/생성한다.
- 실습 노트북은 원본 `.ipynb`를 HTML로 변환해서 오른쪽 패널에 두고, 왼쪽 패널에는 별도 Markdown 실습 가이드를 둔다.
- answer/colab 변형은 특별 지시가 없으면 웹 manifest에 넣지 않는다.
- 자료가 많고 비슷한 장표가 반복될 때는 중복 문장을 늘리지 말고, 같은 개념을 하나의 대표 절로 묶은 뒤 “어느 장표군에서 반복되는 포인트인지”를 적는다.

## 2. 웹 구성 방식

현재 정적 웹은 다음 구조다.

```text
study_notes/                         # 원본 학습 노트 Markdown
study_viewer/                        # GitHub Pages/정적 웹 산출물
  index.html                         # 앱 shell
  styles.css                         # split-view, dark theme, typography
  render.js                          # manifest 로딩, Markdown 렌더링, notebook iframe 연결
  notes-manifest.js                  # window.AI_STUDY_NOTES = [...]
  study_notes/                       # sync 스크립트가 복사한 Markdown
  notebooks/<section>/...html        # ipynb 변환 HTML
scripts/sync_study_viewer.py         # study_notes -> study_viewer 복사 + manifest 생성
scripts/convert_ipynb_static.py      # ipynb -> 안전한 정적 HTML 변환
study_viewer/verify-static-curriculum.js
scripts/browser_smoke_study_viewer.cjs
```

manifest 항목은 두 종류다.

### 일반 Markdown 교안

```js
{
  id: "study-notes-rag-day1-lecture-review",
  title: "RAG Day 1. Retrieval-Augmented Generation 깊은 복습",
  section: "RAG / Day 1",
  path: "study_notes/rag/day1/01_rag_day1_lecture_review.md"
}
```

### 노트북 split-view 항목

```js
{
  id: "rag-day1-practice-01-llama-index",
  kind: "notebook",
  title: "RAG Practice 01. LlamaIndex Query Engine 코드 학습",
  section: "RAG / Day 1 Practice",
  path: "study_notes/rag/practice/01_llama_index_practice_guide.md",
  notebookHtml: "notebooks/rag/day1/01_llama_index.html",
  sourceIpynb: "rag/1일차/실습 자료/Code/1. Llama_index.ipynb"
}
```

`kind: "notebook"`이면 `render.js`가 왼쪽 Markdown + 오른쪽 iframe notebook HTML split view를 만든다.

## 3. ipynb 처리 규칙

1. 원본 ipynb는 그대로 둔다.
2. 왼쪽 가이드는 `study_notes/<topic>/practice/*.md`에 만든다.
3. 오른쪽 노트북 HTML은 다음처럼 생성한다.

```bash
python scripts/convert_ipynb_static.py \
  "rag/1일차/실습 자료/Code/1. Llama_index.ipynb" \
  "study_viewer/notebooks/rag/day1/01_llama_index.html" \
  "RAG Practice 01. LlamaIndex Query Engine"
```

4. 변환 HTML에는 MathJax만 허용된 script로 들어간다. iframe sandbox는 `allow-scripts`만 허용하며 `allow-same-origin`은 넣지 않는다.
5. 실습 가이드는 셀 번호별로 “무엇을 보는지 / 핵심 객체 / shape 또는 데이터 구조 / 실수 포인트”를 적는다.
6. API key가 들어가는 셀은 실행 안내보다 보안 주의와 환경변수 사용을 먼저 적는다.

## 4. 교안 Markdown 작성 스타일

- 첫 줄 제목은 `# <분야> <번호>. <주제> 깊은 복습` 또는 `# <분야> Practice <번호> ... 코드 학습 가이드`로 맞춘다.
- 상단에 대상 원본과 목표를 적는다.
- 가능한 경우 Mermaid 흐름도와 핵심 표를 먼저 둔다.
- 수식은 웹 MathJax가 읽도록 `$...$`, `$$...$$`를 쓴다.
- 코드 중심 자료는 shape-first로 쓴다.
- 개념 중심 자료는 “문제 → 방법 → 왜 필요한가 → 한계 → 실무 체크리스트” 순서로 쓴다.
- 원본 장표가 반복될 때는 반복 장표를 모두 베껴 쓰지 말고 같은 개념을 압축한다.

## 5. 검증 순서

새 항목 추가 후 최소 검증:

```bash
python scripts/sync_study_viewer.py
node study_viewer/verify-static-curriculum.js
```

노트북 HTML을 추가했다면 변환 명령도 먼저 실행한다. UI 변경이 있으면 browser smoke도 실행한다.
