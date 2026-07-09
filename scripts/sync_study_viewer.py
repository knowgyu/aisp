#!/usr/bin/env python3
from __future__ import annotations
import json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'study_notes'
DST = ROOT / 'study_viewer' / 'study_notes'

NOTES = [
  {"id":"study-notes-readme","title":"AISP 복습 노트 인덱스","section":"Overview","path":"study_notes/README.md"},
  {"id":"study-notes-on-device-ai-study-plan","title":"On-Device AI 시험 대비 학습 준비 문맥","section":"Overview","path":"study_notes/on_device_ai_study_plan.md"},
  {"id":"study-notes-on-device-ai-odai1-ch01-on-device-ai-key-points","title":"ODAI-1 Chapter 1. On-Device AI 핵심 정리","section":"On-Device AI / Lecture","path":"study_notes/on_device_ai/odai1_ch01_on_device_ai_key_points.md"},
  {"id":"study-notes-on-device-ai-odai1-ch02-network-pruning-key-points","title":"ODAI-1 Chapter 2. Network Pruning 핵심 정리","section":"On-Device AI / Lecture","path":"study_notes/on_device_ai/odai1_ch02_network_pruning_key_points.md"},
  {"id":"study-notes-on-device-ai-odai1-ch03-quantization-key-points","title":"ODAI-1 Chapter 3. Quantization 핵심 정리","section":"On-Device AI / Lecture","path":"study_notes/on_device_ai/odai1_ch03_quantization_key_points.md"},
  {"id":"study-notes-on-device-ai-odai1-ch04-knowledge-distillation-key-points","title":"ODAI-1 Chapter 4. Knowledge Distillation 핵심 정리","section":"On-Device AI / Lecture","path":"study_notes/on_device_ai/odai1_ch04_knowledge_distillation_key_points.md"},
  {"id":"study-notes-on-device-ai-odai2-ch01-llm-pruning-peft-key-points","title":"ODAI-2 Chapter 1. LLM Pruning & Sparsity-Preserved PEFT 핵심 정리","section":"On-Device AI / Lecture","path":"study_notes/on_device_ai/odai2_ch01_llm_pruning_peft_key_points.md"},
  {"id":"study-notes-on-device-ai-odai2-ch02-llm-quantization-key-points","title":"ODAI-2 Chapter 2. LLM Quantization 핵심 정리","section":"On-Device AI / Lecture","path":"study_notes/on_device_ai/odai2_ch02_llm_quantization_key_points.md"},
  {"id":"study-notes-on-device-ai-odai2-ch03-efficient-inference-key-points","title":"ODAI-2 Chapter 3. Efficient Inference 핵심 정리","section":"On-Device AI / Lecture","path":"study_notes/on_device_ai/odai2_ch03_efficient_inference_key_points.md"},
  {"id":"on-device-practice-01-pruning-cnn","kind":"notebook","title":"Practice 01. Pruning for CNN 코드 학습","section":"On-Device AI / Practice","path":"study_notes/on_device_ai/practice/01_pruning_cnn_practice_guide.md","notebookHtml":"notebooks/on_device_ai/01_pruning_cnn.html","sourceIpynb":"On-Device AI 강의자료/실습/1. Pruning for CNN.ipynb"},
  {"id":"on-device-practice-02-quantization-cnn","kind":"notebook","title":"Practice 02. Quantization for CNN 코드 학습","section":"On-Device AI / Practice","path":"study_notes/on_device_ai/practice/02_quantization_cnn_practice_guide.md","notebookHtml":"notebooks/on_device_ai/02_quantization_cnn.html","sourceIpynb":"On-Device AI 강의자료/실습/2. Quantization for CNN.ipynb"},
  {"id":"on-device-practice-03-knowledge-distillation","kind":"notebook","title":"Practice 03. Knowledge Distillation 코드 학습","section":"On-Device AI / Practice","path":"study_notes/on_device_ai/practice/03_knowledge_distillation_practice_guide.md","notebookHtml":"notebooks/on_device_ai/03_knowledge_distillation.html","sourceIpynb":"On-Device AI 강의자료/실습/3. Knowledge Distillation.ipynb"},
  {"id":"on-device-practice-04-pruning-llm","kind":"notebook","title":"Practice 04. Pruning for LLM 코드 학습","section":"On-Device AI / Practice","path":"study_notes/on_device_ai/practice/04_pruning_llm_practice_guide.md","notebookHtml":"notebooks/on_device_ai/04_pruning_llm.html","sourceIpynb":"On-Device AI 강의자료/실습/4. Pruning for LLM.ipynb"},
  {"id":"on-device-practice-05-quantization-llm","kind":"notebook","title":"Practice 05. Quantization for LLM 코드 학습","section":"On-Device AI / Practice","path":"study_notes/on_device_ai/practice/05_quantization_llm_practice_guide.md","notebookHtml":"notebooks/on_device_ai/05_quantization_llm.html","sourceIpynb":"On-Device AI 강의자료/실습/5. Quantization for LLM.ipynb"},
  {"id":"study-notes-vision-01-resnet18-cifar10-deep-review","title":"Vision 01. ResNet18 + CIFAR-10 깊은 복습","section":"Vision","path":"study_notes/vision/01_resnet18_cifar10_deep_review.md"},
  {"id":"study-notes-vision-02-vit-cifar10-deep-review","title":"Vision 02. Vision Transformer + CIFAR-10 깊은 복습","section":"Vision","path":"study_notes/vision/02_vit_cifar10_deep_review.md"},
  {"id":"study-notes-vision-03-detr-deep-review","title":"Vision 03. DETR 깊은 복습","section":"Vision","path":"study_notes/vision/03_detr_deep_review.md"},
  {"id":"study-notes-vision-04-unet-deep-review","title":"Vision 04. U-Net + Segmentation 깊은 복습","section":"Vision","path":"study_notes/vision/04_unet_deep_review.md"},
  {"id":"study-notes-language-01-vector-space-deep-review","title":"LLM 01. Vector Space / Word Embedding 깊은 복습","section":"Language / LLM","path":"study_notes/language/01_vector_space_deep_review.md"},
  {"id":"study-notes-language-02-dataset-tokenizer-deep-review","title":"LLM 02. Dataset / Tokenizer / Embedding 깊은 복습","section":"Language / LLM","path":"study_notes/language/02_dataset_tokenizer_deep_review.md"},
  {"id":"study-notes-language-03-attention-deep-review","title":"LLM 03. Causal / Multi-Head Attention 깊은 복습","section":"Language / LLM","path":"study_notes/language/03_attention_deep_review.md"},
  {"id":"study-notes-language-04-gpt-architecture-deep-review","title":"LLM 04. GPT Architecture 깊은 복습","section":"Language / LLM","path":"study_notes/language/04_gpt_architecture_deep_review.md"},
  {"id":"study-notes-language-05-pretraining-deep-review","title":"LLM 05. Pretraining 깊은 복습","section":"Language / LLM","path":"study_notes/language/05_pretraining_deep_review.md"},
  {"id":"study-notes-language-06-classification-finetuning-deep-review","title":"LLM 06. Classification Fine-tuning 깊은 복습","section":"Language / LLM","path":"study_notes/language/06_classification_finetuning_deep_review.md"},
  {"id":"study-notes-language-06-lora-finetuning-deep-review","title":"LLM 06+. LoRA Fine-tuning 깊은 복습","section":"Language / LLM","path":"study_notes/language/06_lora_finetuning_deep_review.md"},
  {"id":"study-notes-language-07-instruction-finetuning-deep-review","title":"LLM 07. Instruction Fine-tuning 깊은 복습","section":"Language / LLM","path":"study_notes/language/07_instruction_finetuning_deep_review.md"},
  {"id":"study-notes-language-07-dpo-deep-review","title":"LLM 07+. DPO 깊은 복습 (Optional)","section":"Language / LLM / Optional","path":"study_notes/language/07_dpo_deep_review.md"},

  {"id":"vision-practice-01-resnet18-cifar10","kind":"notebook","title":"Vision Practice 01. ResNet18 CIFAR-10 코드 학습","section":"Vision / Practice","path":"study_notes/vision/practice/01_resnet18_cifar10_practice_guide.md","notebookHtml":"notebooks/vision/01_resnet18_cifar10.html","sourceIpynb":"vision/01_ResNet18_CIFAR10.ipynb"},
  {"id":"vision-practice-02-vit-cifar10","kind":"notebook","title":"Vision Practice 02. ViT CIFAR-10 코드 학습","section":"Vision / Practice","path":"study_notes/vision/practice/02_vit_cifar10_practice_guide.md","notebookHtml":"notebooks/vision/02_vit_cifar10.html","sourceIpynb":"vision/02_ViT_CIFAR10.ipynb"},
  {"id":"vision-practice-03-detr","kind":"notebook","title":"Vision Practice 03. DETR 코드 학습","section":"Vision / Practice","path":"study_notes/vision/practice/03_detr_practice_guide.md","notebookHtml":"notebooks/vision/03_detr.html","sourceIpynb":"vision/03_DETR.ipynb"},
  {"id":"vision-practice-04-unet","kind":"notebook","title":"Vision Practice 04. U-Net Segmentation 코드 학습","section":"Vision / Practice","path":"study_notes/vision/practice/04_unet_practice_guide.md","notebookHtml":"notebooks/vision/04_unet.html","sourceIpynb":"vision/04_Unet.ipynb"},
  {"id":"llm-practice-01-vector-space","kind":"notebook","title":"LLM Practice 01. Vector Space 코드 학습","section":"Language / LLM Practice","path":"study_notes/language/practice/01_vector_space_practice_guide.md","notebookHtml":"notebooks/language/01_vector_space.html","sourceIpynb":"llm_hands_on/Chapter_1_Exercise_Vector Space.ipynb"},
  {"id":"llm-practice-02-dataset","kind":"notebook","title":"LLM Practice 02. Dataset / Tokenizer 코드 학습","section":"Language / LLM Practice","path":"study_notes/language/practice/02_dataset_practice_guide.md","notebookHtml":"notebooks/language/02_dataset.html","sourceIpynb":"llm_hands_on/Chapter_2_Exercise_Dataset.ipynb"},
  {"id":"llm-practice-03-attention","kind":"notebook","title":"LLM Practice 03. Attention 코드 학습","section":"Language / LLM Practice","path":"study_notes/language/practice/03_attention_practice_guide.md","notebookHtml":"notebooks/language/03_attention.html","sourceIpynb":"llm_hands_on/Chapter_3_Excercise_Attention.ipynb"},
  {"id":"llm-practice-03-attention-visualization","kind":"notebook","title":"LLM Practice 03+. Attention Visualization 코드 학습","section":"Language / LLM Practice","path":"study_notes/language/practice/03_attention_visualization_practice_guide.md","notebookHtml":"notebooks/language/03_attention_visualization.html","sourceIpynb":"llm_hands_on/Chapter_3_Excercise_Viz_Multi_head_attention.ipynb"},
  {"id":"llm-practice-04-gpt","kind":"notebook","title":"LLM Practice 04. GPT Architecture 코드 학습","section":"Language / LLM Practice","path":"study_notes/language/practice/04_gpt_practice_guide.md","notebookHtml":"notebooks/language/04_gpt.html","sourceIpynb":"llm_hands_on/Chapter_4_Excercise_GPT.ipynb"},
  {"id":"llm-practice-05-pretraining","kind":"notebook","title":"LLM Practice 05. Pretraining 코드 학습","section":"Language / LLM Practice","path":"study_notes/language/practice/05_pretraining_practice_guide.md","notebookHtml":"notebooks/language/05_pretraining.html","sourceIpynb":"llm_hands_on/Chapter_5_Excercise_Pretraining.ipynb"},
  {"id":"llm-practice-06-classification-finetuning","kind":"notebook","title":"LLM Practice 06. Classification Fine-tuning 코드 학습","section":"Language / LLM Practice","path":"study_notes/language/practice/06_classification_finetuning_practice_guide.md","notebookHtml":"notebooks/language/06_classification_finetuning.html","sourceIpynb":"llm_hands_on/Chapter_6_Excercise_Finetuning_Classification.ipynb"},
  {"id":"llm-practice-06-lora-classification","kind":"notebook","title":"LLM Practice 06+. LoRA Classification 코드 학습","section":"Language / LLM Practice","path":"study_notes/language/practice/06_lora_classification_practice_guide.md","notebookHtml":"notebooks/language/06_lora_classification.html","sourceIpynb":"llm_hands_on/Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb"},
  {"id":"llm-practice-07-instruction-finetuning","kind":"notebook","title":"LLM Practice 07. Instruction Fine-tuning 코드 학습","section":"Language / LLM Practice","path":"study_notes/language/practice/07_instruction_finetuning_practice_guide.md","notebookHtml":"notebooks/language/07_instruction_finetuning.html","sourceIpynb":"llm_hands_on/Chapter_7_Exercise_Follow_Instructions.ipynb"},
  {"id":"llm-practice-07-dpo","kind":"notebook","title":"LLM Practice 07+. DPO 코드 학습","section":"Language / LLM Practice","path":"study_notes/language/practice/07_dpo_practice_guide.md","notebookHtml":"notebooks/language/07_dpo.html","sourceIpynb":"llm_hands_on/Chapter_7_Exercise_Follow_Instructions_dpo.ipynb"},

{"id":"study-notes-rag-day1-lecture-review","title":"RAG Day 1. Retrieval-Augmented Generation 깊은 복습","section":"RAG / Day 1","path":"study_notes/rag/day1/01_rag_day1_lecture_review.md"},
{"id":"rag-day1-practice-01-llama-index","kind":"notebook","title":"RAG Practice 01. LlamaIndex Query Engine 코드 학습","section":"RAG / Day 1 Practice","path":"study_notes/rag/practice/01_llama_index_practice_guide.md","notebookHtml":"notebooks/rag/day1/01_llama_index.html","sourceIpynb":"rag/1일차/실습 자료/Code/1. Llama_index.ipynb"},
{"id":"rag-day1-practice-02-rag-app","kind":"notebook","title":"RAG Practice 02. RAG App 구성 코드 학습","section":"RAG / Day 1 Practice","path":"study_notes/rag/practice/02_rag_practice_guide.md","notebookHtml":"notebooks/rag/day1/02_rag_app.html","sourceIpynb":"rag/1일차/실습 자료/Code/2. RAG.ipynb"},
]

DENIED = [
  'study_notes/on_device_ai/ch01_pdf_extracted_text.md',
  'study_notes/on_device_ai/ch01_lecture_pack.md',
  'study_notes/on_device_ai/ch01_lecture_pack.html',
  'study_notes/on_device_ai/01_cnn_pruning_deep_review.md',
]

def image_refs(markdown: str):
    import re
    for match in re.finditer(r'!\[[^\]]*\]\(([^)]+)\)', markdown):
        target = match.group(1).split('#')[0].strip()
        if not target or target.startswith(('http:', 'https:', 'data:', '#')):
            continue
        yield target


def copy_referenced_assets(note_rel: str):
    src_note = ROOT / note_rel
    dst_note = ROOT / 'study_viewer' / note_rel
    for ref in image_refs(src_note.read_text(encoding='utf-8')):
        src_asset = (src_note.parent / ref).resolve()
        dst_asset = (dst_note.parent / ref).resolve()
        if not src_asset.exists():
            raise FileNotFoundError(f'missing referenced asset {ref} from {note_rel}: {src_asset}')
        # Keep copied assets inside the Pages artifact.
        if not str(dst_asset).startswith(str((ROOT / 'study_viewer').resolve())):
            raise RuntimeError(f'referenced asset would escape public artifact: {note_rel} -> {ref}')
        dst_asset.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_asset, dst_asset)


def copy_allowed():
    if DST.exists():
        shutil.rmtree(DST)
    # Remove generated external-asset mirrors that may have been copied by an
    # older sync pass. The script will recreate only assets referenced by
    # allow-listed Markdown.
    for stale in [ROOT / 'study_viewer' / 'llm_lecture2']:
        if stale.exists():
            shutil.rmtree(stale)
    for note in NOTES:
        rel = note['path']
        src = ROOT / rel
        dst = ROOT / 'study_viewer' / rel
        if not src.exists():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copy_referenced_assets(rel)
    for denied in DENIED:
        p = ROOT / 'study_viewer' / denied
        if p.exists():
            raise RuntimeError(f'denied file still in public artifact: {p}')
    stale_ch01_assets = ROOT / 'study_viewer' / 'study_notes' / 'on_device_ai' / 'assets' / 'ch01_cnn_pruning_slides'
    if stale_ch01_assets.exists():
        raise RuntimeError(f'denied ch01 asset directory still in public artifact: {stale_ch01_assets}')


def write_manifest():
    out = 'window.AI_STUDY_NOTES = ' + json.dumps(NOTES, ensure_ascii=False, indent=2) + ';\n'
    (ROOT / 'study_viewer' / 'notes-manifest.js').write_text(out, encoding='utf-8')

if __name__ == '__main__':
    copy_allowed()
    write_manifest()
    print(f'synced {len(NOTES)} public notes')
