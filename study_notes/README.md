# AISP 복습 노트 인덱스

기존 교안/노트북은 그대로 두고, 복습용 상세 MD만 `study_notes/` 아래에 추가했습니다.

## Vision: Day3/Day4 복습

| 순서 | 복습 노트 | 원본 |
|---:|---|---|
| 01 | [ResNet18 + CIFAR-10](vision/01_resnet18_cifar10_deep_review.md) | `vision/01_Intro.pdf`, `vision/02_DNN_CNN.pdf`, `vision/01_ResNet18_CIFAR10.ipynb` |
| 02 | [Vision Transformer + CIFAR-10](vision/02_vit_cifar10_deep_review.md) | `vision/03_Attention.pdf`, `vision/02_ViT_CIFAR10.ipynb` |
| 03 | [DETR](vision/03_detr_deep_review.md) | `vision/03_Attention.pdf`, `vision/04_Applications.pdf`, `vision/03_DETR.ipynb` |
| 04 | [U-Net](vision/04_unet_deep_review.md) | `vision/04_Applications.pdf`, `vision/04_Unet.ipynb` |

## Language / LLM 복습

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
| 07+ | [DPO](language/07_dpo_deep_review.md) | `llm_hands_on/Chapter_7_Exercise_Follow_Instructions_dpo.ipynb` |

## 추천 복습 순서

1. Vision은 `01 -> 02 -> 03 -> 04` 순서로 보세요. CNN feature map에서 token sequence, detection, segmentation으로 자연스럽게 이어집니다.
2. LLM은 `02 -> 03 -> 04 -> 05 -> 06/07` 순서가 코드 이해에 가장 좋습니다. 01은 embedding 직관용 워밍업입니다.
