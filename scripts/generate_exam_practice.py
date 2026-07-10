#!/usr/bin/env python3
"""Generate fill-in practice notebooks and answer guides without editing originals."""
from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path

import sync_study_viewer

ROOT = Path(__file__).resolve().parents[1]
PRACTICE_ROOT = ROOT / "practice_notebooks"
ANSWER_ROOT = ROOT / "study_notes" / "exam_answers"
PRACTICE_NOTICE = "<!-- aisp-exam-practice-notice -->"


@dataclass(frozen=True)
class NotebookSpec:
    id: str
    title: str
    subject: str
    source: str
    output: str
    explicit_cells: tuple[int, ...] = ()


SUBJECT_LABELS = {
    "vision": "Vision",
    "language": "Language / LLM",
    "on_device_ai": "On-Device AI",
    "rag": "RAG",
}

SUBJECT_TERMS = {
    "vision": (
        "conv", "residual", "forward", "transform", "patch", "attention",
        "encoder", "decoder", "loss", "optimizer", "dataloader", "mask",
    ),
    "language": (
        "token", "embedding", "attention", "query", "key", "value", "mask",
        "forward", "loss", "optimizer", "dataloader", "lora", "dpo", "gpt",
    ),
    "on_device_ai": (
        "prun", "mask", "quant", "scale", "zero_point", "distill", "teacher",
        "student", "loss", "forward", "perplex", "wanda", "calibr",
    ),
    "rag": (
        "document", "chunk", "embedding", "index", "retrieve", "query_engine",
        "reader", "prompt", "rag", "mcp", "tool", "resource", "kg", "eval",
    ),
}

RAG_DAY2 = (
    NotebookSpec(
        "rag-day2-practice-01-mcp-evaluation",
        "RAG Day 2 Practice 01. MCP 기반 평가와 업그레이드",
        "rag",
        "rag/2일차/실습 자료/Code/4_RAG_framework_evaluation_with_MCP.ipynb",
        "practice_notebooks/rag/day2/01_mcp_evaluation.ipynb",
        (17, 19, 22, 46, 55),
    ),
    NotebookSpec(
        "rag-day2-practice-02-data-preprocessing",
        "RAG Day 2 Practice 02. CRAG 데이터 전처리",
        "rag",
        "rag/2일차/실습 자료/Code/1. Data_preprocessing.ipynb",
        "practice_notebooks/rag/day2/02_data_preprocessing.ipynb",
        (16, 19, 21, 44),
    ),
    NotebookSpec(
        "rag-day2-practice-03-web-rag",
        "RAG Day 2 Practice 03. Web Retriever와 Reader",
        "rag",
        "rag/2일차/실습 자료/Code/2. Task_1.ipynb",
        "practice_notebooks/rag/day2/03_web_rag.ipynb",
        (7, 12, 16, 23, 25, 29),
    ),
    NotebookSpec(
        "rag-day2-practice-04-kg-rag",
        "RAG Day 2 Practice 04. Knowledge Graph RAG",
        "rag",
        "rag/2일차/실습 자료/Code/3. Task_2.ipynb",
        "practice_notebooks/rag/day2/04_kg_rag.ipynb",
        (13, 15, 21, 25, 31, 36),
    ),
)


def subject_for(note: dict[str, str]) -> str:
    path = note.get("referenceIpynb", note["sourceIpynb"])
    if path.startswith("vision/"):
        return "vision"
    if path.startswith("llm_hands_on/"):
        return "language"
    if path.startswith("On-Device AI"):
        return "on_device_ai"
    if path.startswith("rag/"):
        return "rag"
    raise ValueError(f"unknown notebook subject: {path}")


def slug_from_id(note_id: str) -> str:
    return re.sub(r"^(?:on-device|vision|llm|rag-day1)-practice-", "", note_id)


def existing_specs() -> list[NotebookSpec]:
    specs = []
    for note in sync_study_viewer.NOTES:
        if note.get("kind") != "notebook":
            continue
        if note["id"].startswith("rag-day2-"):
            continue
        subject = subject_for(note)
        source = note.get("referenceIpynb", note["sourceIpynb"])
        day = "/day1" if source.startswith("rag/1일차/") else ""
        specs.append(
            NotebookSpec(
                note["id"],
                note["title"],
                subject,
                source,
                f"practice_notebooks/{subject}{day}/{slug_from_id(note['id'])}.ipynb",
            )
        )
    return specs


def all_specs() -> list[NotebookSpec]:
    return [*existing_specs(), *RAG_DAY2]


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source or "")


def write_text_if_changed(path: Path, content: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def previous_heading(cells: list[dict], index: int) -> str:
    for cell in reversed(cells[:index]):
        if cell.get("cell_type") != "markdown":
            continue
        headings = [
            re.sub(r"^#+\s*", "", line).strip()
            for line in source_text(cell).splitlines()
            if re.match(r"^#{1,6}\s*\S", line.strip())
        ]
        if headings:
            return headings[-1][:120]
    return "핵심 코드 골격"


def score_code(source: str, subject: str) -> int:
    lowered = source.lower()
    lines = [line for line in source.splitlines() if line.strip()]
    if len(lines) < 3:
        return -100
    score = min(len(lines) // 8, 4)
    score += 8 * len(re.findall(r"^\s*class\s+", source, re.M))
    score += 6 * len(re.findall(r"^\s*(?:async\s+)?def\s+", source, re.M))
    score += 3 if "forward(" in lowered else 0
    score += 2 if "### your code" in lowered else 0
    score += sum(2 for term in SUBJECT_TERMS[subject] if term in lowered)
    if re.search(r"pip install|wget |curl |openai_api_key|os\.environ", lowered):
        score -= 8
    if all(line.lstrip().startswith(("import ", "from ", "%", "!")) for line in lines):
        score -= 10
    if len(lines) <= 5 and re.search(r"\b(print|display|plot|show)\s*\(", lowered):
        score -= 6
    return score


def select_cells(spec: NotebookSpec, cells: list[dict]) -> list[int]:
    if spec.explicit_cells:
        selected = list(spec.explicit_cells)
    else:
        candidates = []
        for index, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue
            score = score_code(source_text(cell), spec.subject)
            if score > 2:
                candidates.append((score, index))
        limit = 5 if len(cells) >= 35 else 4
        selected = [index for _, index in sorted(candidates, key=lambda item: (-item[0], item[1]))[:limit]]
        selected.sort()
    for index in selected:
        if index >= len(cells) or cells[index].get("cell_type") != "code":
            raise ValueError(f"{spec.source}: invalid practice cell {index}")
    if not selected:
        raise ValueError(f"{spec.source}: no practice cells selected")
    return selected


def practice_intro(spec: NotebookSpec, selected: list[int]) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {"aisp_exam_practice": True},
        "source": [
            f"# {spec.title} — 시험 대비 실습본\n",
            "\n",
            f"- 원본: `{spec.source}`\n",
            f"- 핵심 빈칸 수: **{len(selected)}개**\n",
            "- `## 정답 입력` 셀을 직접 구현한 뒤 과목별 정답·해설지와 비교하세요.\n",
            "- API key, 외부 서버 주소, 대용량 데이터 경로는 자신의 환경에 맞게 설정하세요.\n",
        ],
    }


def blank_cell(spec: NotebookSpec, cells: list[dict], index: int, drill_number: int) -> dict:
    heading = previous_heading(cells, index)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "aisp_exam_practice": True,
            "source_cell_index": index,
            "drill_number": drill_number,
        },
        "outputs": [],
        "source": [
            "# ## 정답 입력\n",
            f"# Drill {drill_number}: {heading}\n",
            f"# 원본 Cell {index:03d}의 핵심 코드를 직접 작성하세요.\n",
            "# 과목별 정답·해설지에는 원본 코드와 출제 의도가 있습니다.\n",
            "\n",
            "pass\n",
        ],
    }


def generate_notebook(spec: NotebookSpec) -> tuple[list[dict], list[int]]:
    source_path = ROOT / spec.source
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    original_cells = notebook.get("cells", [])
    selected = select_cells(spec, original_cells)
    selected_set = set(selected)
    output_cells = [practice_intro(spec, selected)]
    drill_number = 0
    for index, cell in enumerate(original_cells):
        if index in selected_set:
            drill_number += 1
            output_cells.append(blank_cell(spec, original_cells, index, drill_number))
        else:
            cloned = copy.deepcopy(cell)
            if cloned.get("cell_type") == "code":
                cloned["execution_count"] = None
                cloned["outputs"] = []
            output_cells.append(cloned)
    notebook["cells"] = output_cells
    notebook.setdefault("metadata", {})["aisp_exam_practice"] = {
        "source": spec.source,
        "selected_cells": selected,
        "generated_by": "scripts/generate_exam_practice.py",
    }
    output_path = ROOT / spec.output
    write_text_if_changed(output_path, json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    return original_cells, selected


def ensure_guide_notice(specs: list[NotebookSpec]) -> int:
    notes_by_id = {note["id"]: note for note in sync_study_viewer.NOTES}
    changed = 0
    for spec in specs:
        note = notes_by_id.get(spec.id)
        if not note or spec.id.startswith("rag-day2-"):
            continue
        guide_path = ROOT / note["path"]
        text = guide_path.read_text(encoding="utf-8")
        if PRACTICE_NOTICE in text:
            continue
        lines = text.splitlines()
        insert_at = 1 if lines and lines[0].startswith("#") else 0
        notice = [
            "",
            PRACTICE_NOTICE,
            "> **시험 대비 모드:** 오른쪽에는 원본 전체 코드 대신 핵심 셀을 `## 정답 입력`으로 비운 실습본이 표시됩니다.",
            f"> 원본은 `{spec.source}`, 정답과 출제 의도는 `{ANSWER_ROOT.relative_to(ROOT) / f'{spec.subject}_code_answers.md'}`에서 확인합니다.",
            "",
        ]
        lines[insert_at:insert_at] = notice
        guide_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        changed += 1
    return changed


def answer_document(subject: str, entries: list[tuple[NotebookSpec, list[dict], list[int]]]) -> str:
    label = SUBJECT_LABELS[subject]
    out = [
        f"# {label} 코드 실습 정답·해설지",
        "",
        "이 문서는 생성된 시험 대비 실습 노트북의 `## 정답 입력` 셀에 대응한다.",
        "정답을 바로 복사하기보다 먼저 입력·실행하고, 실패 원인을 기록한 뒤 비교한다.",
        "",
        "## 출제 포인트 기준",
        "",
    ]
    if subject == "rag":
        out += [
            "- RAG 구성요소와 `retrieve → augment → generate` 흐름",
            "- LlamaIndex, MCP 등 관련 라이브러리의 역할과 연결",
            "- Retriever/Reader/KG/MCP tool 호출의 입력·출력 구조",
        ]
    elif subject == "vision":
        out += [
            "- 주요 layer 구성과 `[B, C, H, W]` tensor 흐름",
            "- 모델 출력과 정답을 loss/metric으로 연결하는 과정",
            "- 이미지 transform과 학습·평가 데이터 구성",
        ]
    elif subject == "language":
        out += [
            "- embedding, attention, GPT layer의 구성과 tensor 흐름",
            "- next-token/classification/instruction 학습의 loss 연결",
            "- tokenizer와 dataset이 모델 입력을 만드는 과정",
        ]
    else:
        out += [
            "- Quantization의 scale/zero-point와 양자화·역양자화",
            "- Pruning 중요도·mask와 연산 제거",
            "- Teacher/Student 출력과 정답을 결합한 distillation loss",
        ]
    out.append("")
    for spec, cells, selected in entries:
        out += [f"## {spec.title}", "", f"원본: `{spec.source}`", f"실습본: `{spec.output}`", ""]
        for drill_number, index in enumerate(selected, 1):
            heading = previous_heading(cells, index)
            source = "\n".join(
                line.rstrip() for line in source_text(cells[index]).rstrip().splitlines()
            )
            out += [
                f"### Drill {drill_number} — {heading}",
                "",
                f"원본 Cell `{index:03d}`. 이 셀은 **{heading}** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.",
                "",
                "```python",
                source,
                "```",
                "",
                "**확인 질문**",
                "",
                "1. 이 셀의 입력 객체와 출력 객체는 무엇인가?",
                "2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?",
                "3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?",
                "",
            ]
    return "\n".join(out).rstrip() + "\n"


def generate_all() -> dict[str, int]:
    grouped: dict[str, list[tuple[NotebookSpec, list[dict], list[int]]]] = {
        subject: [] for subject in SUBJECT_LABELS
    }
    drill_count = 0
    specs = all_specs()
    for spec in specs:
        cells, selected = generate_notebook(spec)
        grouped[spec.subject].append((spec, cells, selected))
        drill_count += len(selected)
    ANSWER_ROOT.mkdir(parents=True, exist_ok=True)
    for subject, entries in grouped.items():
        write_text_if_changed(
            ANSWER_ROOT / f"{subject}_code_answers.md",
            answer_document(subject, entries),
        )
    changed_guides = ensure_guide_notice(specs)
    return {
        "notebooks": len(specs),
        "drills": drill_count,
        "answer_guides": len(grouped),
        "guide_notices": changed_guides,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print configured notebooks without writing")
    args = parser.parse_args()
    if args.list:
        for spec in all_specs():
            print(f"{spec.subject}\t{spec.source}\t{spec.output}")
        return 0
    stats = generate_all()
    print(
        "generated exam practice: "
        f"notebooks={stats['notebooks']} drills={stats['drills']} "
        f"answer_guides={stats['answer_guides']} guide_notices={stats['guide_notices']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
