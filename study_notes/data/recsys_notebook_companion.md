# 추천시스템 Notebook Companion: GCF와 NCF shape/metric

대상 원본: `data/7_8-recsys-practice/RecSys_GCF_practice.ipynb`, `RecSys_GCF_sol.ipynb`, `RecSys_NCF.ipynb`, `study_viewer/notebooks/data/`
관련 PDF: `data/5-recsys-basics.pdf`, `data/6-recsys-advanced.pdf`, `data/7_8-recsys-practice.pdf`

## 1. 상호작용에서 score까지

```text
(user_id,item_id,rating/click) -> contiguous ID mapping
-> user/item embedding -> model score -> loss -> top-K ranking
```

- mini-batch user/item ID: `[B]` (dtype `torch.long`).
- embedding table: user `[U,D]`, item `[I,D]`; lookup 결과 `[B,D]`.
- MF/GCF dot product: `(u * v).sum(dim=1)` → `[B]`; rating head가 있으면 `[B,1]`.
- NCF: concat `[u_emb,v_emb]` → `[B,2D]` → MLP → `[B,1]` → `squeeze(-1)` → `[B]`.

## 2. GCF와 NCF 역할

| 모델 | 핵심 | shape 흐름 |
|---|---|---|
| MF | user/item latent dot product | `[B,D] × [B,D] → [B]` |
| GCF | user-item graph message propagation | node `[U+I,D]` → normalized adjacency → `[U+I,D]` → pair score `[B]` |
| NCF | interaction을 MLP로 학습 | `[B,D] + [B,D] → [B,2D] → [B,1]` |

GCF에서 adjacency 정규화와 self-loop, user/item node indexing이 일관되어야 한다. NCF에서는 `Embedding` 입력 dtype과 output squeeze 축을 확인한다.

## 3. loss와 평가 API

- explicit rating: `MSELoss`/`L1Loss`, RMSE/MAE. 예측 `[B]`, target `[B]`를 맞춘다.
- implicit feedback: BCE with logits 또는 pairwise BPR. positive item score가 negative item보다 높아지도록 학습한다.
- ranking: train에서 본 item을 평가 후보에서 제외한 뒤 `torch.topk(scores,K)`를 적용한다.
- `Precision@K = hits/K`, `Recall@K = hits/|relevant|`, `NDCG@K`는 순위 discount를 적용한다.

rating RMSE가 좋다고 recommendation top-K가 좋은 것은 아니다. 목적이 별도이므로 두 metric을 함께 보고, user별 cold-start/빈도 편향을 점검한다.

## 4. 빈칸/오류 체크

1. ID를 `0..U-1`, `0..I-1`로 mapping했는가?
2. embedding 입력이 `LongTensor`인가?
3. negative sampling이 positive와 같은 item을 뽑지 않는가?
4. train interaction을 validation 추천 결과에 그대로 포함하지 않는가?
5. `squeeze()`가 batch size 1일 때 batch 축까지 제거하지 않는가? `squeeze(-1)`을 선호한다.
6. graph propagation에서 adjacency shape가 `[U+I,U+I]`인지 확인했는가?

## 5. 시험 답안 한 줄

GCF는 user-item graph의 이웃 정보를 embedding에 전파해 협업 신호를 누적하고, NCF는 user/item embedding 상호작용을 MLP로 비선형 학습한다. 둘 다 최종적으로 사용자-아이템 score를 만들지만 propagation/interaction 함수와 평가 목적을 구분해야 한다.

## 6. 원본 notebook의 수치 앵커

- 데이터: MovieLens latest-small `ratings.csv`; NCF는 `LabelEncoder`로 user/item을 연속 ID로 만들고 `random_state=42`로 9:1 분할한다.
- NCF: user/item embedding 각각 32차원 → concat `[B,64]` → `Linear(64,32)+ReLU` → `Linear(32,1)`, `MSELoss`, Adam `0.001`, batch 128, 5 epochs. relevance threshold는 `rating >= 3.5`다.
- GCF/NGCF: rating `>=1.0`을 implicit edge로 변환하며 item node index는 `movieId + num_users`; edge column split은 80/10/10이다.
- NGCF는 초기 node embedding 64차원, `layer_dims=[64,64]`, degree normalization `1/sqrt(deg(u)*deg(i))`, self/neighbor message와 LeakyReLU/dropout을 사용한다. 학습 objective는 positive/negative item을 비교하는 BPR loss이고 Recall/Precision/NDCG@10으로 평가한다.
