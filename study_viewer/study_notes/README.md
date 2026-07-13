# AISP 복습 노트 인덱스

이 폴더는 원본 강의자료/PDF/ipynb를 그대로 대체하지 않고, 시험 준비용으로 다시 읽기 쉽게 정리한 overlay 노트입니다.


## 시험 직전 빠른 진입점

- [Jupyter Skeleton Code 생존 체크리스트](jupyter_exam_quick_reference.md) — 실행 순서, shape, training/eval, 누수 점검
- [Data 큰 그림·PDF 링크·출제 포인트](data/00_exam_data_overview.md)
- [RAG Day 1 코드 동반 복습](rag/day1/02_rag_day1_code_companion.md)
- [RAG Day 2 CRAG·KG·MCP 코드 동반 복습](rag/day2/03_crag_kg_mcp_notebook_companion.md)
- [Time Series notebook 동반 복습](data/04_time_series_notebook_companion.md)
- [Recommender notebook 동반 복습](data/05_recommender_notebook_companion.md)

## 시험 Recap 학습 계획

실습 빈칸을 풀기 전에 출제포인트와 원본 교안·노트북을 순서대로 복습하려면 [통합 Recap 계획](recap/00_master_recap_plan.md)부터 시작합니다.

권장 순서는 **LLM → Vision → On-device AI → RAG**이며, 각 과목의 상세 파일은 다음과 같습니다.

- [LLM Recap](recap/01_llm_recap_plan.md)
- [Vision Recap](recap/02_vision_recap_plan.md)
- [On-device AI Recap](recap/03_on_device_ai_recap_plan.md)
- [RAG Recap](recap/04_rag_recap_plan.md)

Data는 별도 교안/실습을 학습하기 전까지 보류합니다.

## On-Device AI — Lecture notes

| 순서 | 복습 노트 | 원본 |
|---:|---|---|
| ODAI-1 Ch.1 | [On-Device AI 핵심 정리](on_device_ai/odai1_ch01_on_device_ai_key_points.md) | `On-Device AI 강의자료/ODAI-1.pdf` |
| ODAI-1 Ch.2 | [Network Pruning 핵심 정리](on_device_ai/odai1_ch02_network_pruning_key_points.md) | `On-Device AI 강의자료/ODAI-1.pdf` |
| ODAI-1 Ch.3 | [Quantization 핵심 정리](on_device_ai/odai1_ch03_quantization_key_points.md) | `On-Device AI 강의자료/ODAI-1.pdf` |
| ODAI-1 Ch.4 | [Knowledge Distillation 핵심 정리](on_device_ai/odai1_ch04_knowledge_distillation_key_points.md) | `On-Device AI 강의자료/ODAI-1.pdf` |
| ODAI-2 Ch.1 | [LLM Pruning & Sparsity-Preserved PEFT](on_device_ai/odai2_ch01_llm_pruning_peft_key_points.md) | `On-Device AI 강의자료/ODAI-2.pdf` |
| ODAI-2 Ch.2 | [LLM Quantization](on_device_ai/odai2_ch02_llm_quantization_key_points.md) | `On-Device AI 강의자료/ODAI-2.pdf` |
| ODAI-2 Ch.3 | [Efficient Inference](on_device_ai/odai2_ch03_efficient_inference_key_points.md) | `On-Device AI 강의자료/ODAI-2.pdf` |

## On-Device AI — Practice notebooks

웹 viewer에서는 아래 가이드와 핵심 셀을 `## 정답 입력`으로 비운 시험 대비 노트북 HTML이 split view로 동시에 표시됩니다. 원본은 변경하지 않습니다.

| 순서 | 실습 가이드 | 원본 notebook |
|---:|---|---|
| 01 | [Pruning for CNN](on_device_ai/practice/01_pruning_cnn_practice_guide.md) | `On-Device AI 강의자료/실습/1. Pruning for CNN.ipynb` |
| 02 | [Quantization for CNN](on_device_ai/practice/02_quantization_cnn_practice_guide.md) | `On-Device AI 강의자료/실습/2. Quantization for CNN.ipynb` |
| 03 | [Knowledge Distillation](on_device_ai/practice/03_knowledge_distillation_practice_guide.md) | `On-Device AI 강의자료/실습/3. Knowledge Distillation.ipynb` |
| 04 | [Pruning for LLM](on_device_ai/practice/04_pruning_llm_practice_guide.md) | `On-Device AI 강의자료/실습/4. Pruning for LLM.ipynb` |
| 05 | [Quantization for LLM](on_device_ai/practice/05_quantization_llm_practice_guide.md) | `On-Device AI 강의자료/실습/5. Quantization for LLM.ipynb` |

## Vision

| 순서 | 복습 노트 | 원본 |
|---:|---|---|
| 01 | [ResNet18 + CIFAR-10](vision/01_resnet18_cifar10_deep_review.md) | `vision/01_Intro.pdf`, `vision/02_DNN_CNN.pdf`, `vision/01_ResNet18_CIFAR10.ipynb` |
| 02 | [Vision Transformer + CIFAR-10](vision/02_vit_cifar10_deep_review.md) | `vision/03_Attention.pdf`, `vision/02_ViT_CIFAR10.ipynb` |
| 03 | [DETR](vision/03_detr_deep_review.md) | `vision/03_Attention.pdf`, `vision/04_Applications.pdf`, `vision/03_DETR.ipynb` |
| 04 | [U-Net](vision/04_unet_deep_review.md) | `vision/04_Applications.pdf`, `vision/04_Unet.ipynb` |

Vision 05+ generative material은 이번 시험 준비 public viewer 범위에서 제외했습니다.

## Language / LLM

| 순서 | 복습 노트 | 원본 |
|---:|---|---|
| 01 | [Vector Space / Word Embedding](language/01_vector_space_deep_review.md) | `llm_hands_on/Chapter_1_Exercise_Vector Space.ipynb` |
| 02 | [Dataset / Tokenizer / Embedding](language/02_dataset_tokenizer_deep_review.md) | `llm_hands_on/Chapter_2_Exercise_Dataset.ipynb` |
| 03 | [Causal / Multi-Head Attention](language/03_attention_deep_review.md) | `llm_hands_on/Chapter_3_Excercise_Attention.ipynb`, viz notebook |
| 04 | [GPT Architecture](language/04_gpt_architecture_deep_review.md) | `llm_hands_on/Chapter_4_Excercise_GPT.ipynb` |
| 05 | [Pretraining](language/05_pretraining_deep_review.md) | `llm_hands_on/Chapter_5_Excercise_Pretraining.ipynb` |
| 06 | [Classification Fine-tuning](language/06_classification_finetuning_deep_review.md) | `llm_hands_on/Chapter_6_Excercise_Finetuning_Classification.ipynb` |
| 06+ | [LoRA Fine-tuning](language/06_lora_finetuning_deep_review.md) | `llm_hands_on/Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb` |
| 07 | [Instruction Fine-tuning](language/07_instruction_finetuning_deep_review.md) | `llm_hands_on/Chapter_7_Exercise_Follow_Instructions.ipynb` |
| 07+ | [DPO — optional](language/07_dpo_deep_review.md) | `llm_hands_on/Chapter_7_Exercise_Follow_Instructions_dpo.ipynb` |

## RAG — Day 1

| 순서 | 복습 노트 / 실습 가이드 | 원본 |
|---:|---|---|
| 01 | [RAG Day 1 Lecture Review](rag/day1/01_rag_day1_lecture_review.md) | `rag/1일차/` PDF/HTML 자료 |
| Practice 01 | [LlamaIndex Query Engine](rag/practice/01_llama_index_practice_guide.md) | `rag/1일차/실습 자료/Code/1. Llama_index.ipynb` |
| Practice 02 | [RAG App Configuration](rag/practice/02_rag_practice_guide.md) | `rag/1일차/실습 자료/Code/2. RAG.ipynb` |

## RAG — Day 2 (MCP → GraphRAG)

| 순서 | 복습 노트 / 실습 가이드 | 원본 |
|---:|---|---|
| 01 | [MCP와 Text2SQL 깊은 복습](rag/day2/01_mcp_text2sql_review.md) | `rag/2일차/NL2SQL___mcp.pdf` |
| Practice 01 | [MCP 기반 평가와 업그레이드](rag/day2/practice/01_mcp_evaluation_practice_guide.md) | `4_RAG_framework_evaluation_with_MCP.ipynb` |
| 02 | [GraphRAG와 CRAG 깊은 복습](rag/day2/02_graphrag_crag_review.md) | `Graph_RAG.pdf`, Graph RAG 실습 PDF |
| Practice 02 | [CRAG 데이터 전처리](rag/day2/practice/02_data_preprocessing_practice_guide.md) | `1. Data_preprocessing.ipynb` |
| Practice 03 | [Web Retriever와 Reader](rag/day2/practice/03_web_rag_practice_guide.md) | `2. Task_1.ipynb` |
| Practice 04 | [Knowledge Graph RAG](rag/day2/practice/04_kg_rag_practice_guide.md) | `3. Task_2.ipynb` |

## 4과목 코드 실습 정답·해설

- [Vision 정답·해설](exam_answers/vision_code_answers.md)
- [Language / LLM 정답·해설](exam_answers/language_code_answers.md)
- [On-Device AI 정답·해설](exam_answers/on_device_ai_code_answers.md)
- [RAG 정답·해설](exam_answers/rag_code_answers.md)

실습본은 `practice_notebooks/`에 생성되며, `python scripts/generate_exam_practice.py`로 원본에서 다시 만들 수 있습니다.

## Vision — Notebook practice guides

| 순서 | 실습 가이드 | 원본 notebook |
|---:|---|---|
| Practice 01 | [ResNet18 CIFAR-10 코드 학습](vision/practice/01_resnet18_cifar10_practice_guide.md) | `vision/01_ResNet18_CIFAR10.ipynb` |
| Practice 02 | [ViT CIFAR-10 코드 학습](vision/practice/02_vit_cifar10_practice_guide.md) | `vision/02_ViT_CIFAR10.ipynb` |
| Practice 03 | [DETR 코드 학습](vision/practice/03_detr_practice_guide.md) | `vision/03_DETR.ipynb` |
| Practice 04 | [U-Net Segmentation 코드 학습](vision/practice/04_unet_practice_guide.md) | `vision/04_Unet.ipynb` |

> Vision 05 DDPM, 06 Stable Diffusion 등 generative/GAN 계열은 이번 범위에서 제외했습니다.

## LLM — Notebook practice guides

| 순서 | 실습 가이드 | 원본 notebook |
|---:|---|---|
| Practice 01 | [Vector Space 코드 학습](language/practice/01_vector_space_practice_guide.md) | `llm_hands_on/Chapter_1_Exercise_Vector Space.ipynb` |
| Practice 02 | [Dataset / Tokenizer 코드 학습](language/practice/02_dataset_practice_guide.md) | `llm_hands_on/Chapter_2_Exercise_Dataset.ipynb` |
| Practice 03 | [Attention 코드 학습](language/practice/03_attention_practice_guide.md) | `llm_hands_on/Chapter_3_Excercise_Attention.ipynb` |
| Practice 03+ | [Attention Visualization 코드 학습](language/practice/03_attention_visualization_practice_guide.md) | `llm_hands_on/Chapter_3_Excercise_Viz_Multi_head_attention.ipynb` |
| Practice 04 | [GPT Architecture 코드 학습](language/practice/04_gpt_practice_guide.md) | `llm_hands_on/Chapter_4_Excercise_GPT.ipynb` |
| Practice 05 | [Pretraining 코드 학습](language/practice/05_pretraining_practice_guide.md) | `llm_hands_on/Chapter_5_Excercise_Pretraining.ipynb` |
| Practice 06 | [Classification Fine-tuning 코드 학습](language/practice/06_classification_finetuning_practice_guide.md) | `llm_hands_on/Chapter_6_Excercise_Finetuning_Classification.ipynb` |
| Practice 06+ | [LoRA Classification 코드 학습](language/practice/06_lora_classification_practice_guide.md) | `llm_hands_on/Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb` |
| Practice 07 | [Instruction Fine-tuning 코드 학습](language/practice/07_instruction_finetuning_practice_guide.md) | `llm_hands_on/Chapter_7_Exercise_Follow_Instructions.ipynb` |
| Practice 07+ | [DPO 코드 학습](language/practice/07_dpo_practice_guide.md) | `llm_hands_on/Chapter_7_Exercise_Follow_Instructions_dpo.ipynb` |
