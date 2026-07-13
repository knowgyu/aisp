# Recommender System 07–08. NCF·NGCF/GCF notebook 동반 복습

대상 원본: `data/5-recsys-basics.pdf`, `data/6-recsys-advanced.pdf`, `data/7_8-recsys-practice/RecSys_NCF.ipynb`, `RecSys_GCF_practice.ipynb`, MovieLens CSV. 원본과 solution notebook은 변경하지 않는다.

## NCF: 평점 회귀 흐름

```text
ratings.csv → LabelEncoder(userId/movieId) → 9:1 train/test
→ MovieLens Dataset → DataLoader(B=128)
→ user/movie Embedding(32) concat(64) → ReLU MLP → rating
→ MSELoss → RMSE/MAE
```

| 객체 | 역할 | shape |
|---|---|---|
| `MovieLens` | user/movie/rating 샘플 반환 | 각 field `(B,)` |
| `user_embedding` | user latent factor | `(n_users,32)` |
| `movie_embedding` | item latent factor | `(n_movies,32)` |
| concat | 두 latent vector 결합 | `(B,64)` |
| NCF MLP output | 예측 평점 | `(B,1)` |

`LabelEncoder`는 원래 ID를 연속 인덱스로 바꾼다. train/test에 별도 fit하지 말고 train 기준 encoder를 유지한다. `ground_truth`를 `(B,1)`로 reshape해 prediction과 loss shape를 맞춘다.

## NGCF/GCF: 이분 그래프 흐름

```text
rating >= threshold → edge_index(user, num_users+item)
→ edge split(train/val/test) → Embedding(N_user+N_item, d)
→ neighbor message passing → user/item features
→ dot-product score → Top-k metrics
```

| 구조 | shape/규칙 |
|---|---|
| `edge_index` | `(2,E)`; column 하나가 `(src,dst)` |
| node embedding `H0` | `(num_users+num_items, embedding_dim)` |
| user node | index `0..num_users-1` |
| item node | index `num_users..num_users+num_items-1` |
| output user/item feature | `(num_users,d)`, `(num_items,d)` |
| score | user vector와 item vector 내적, `(num_items,)` per user |

item index offset을 제거할 때 `item_id = edge[1] - num_users`를 잊지 않는다. adjacency matrix 대신 sparse `edge_index`를 사용하면 GNN message passing이 가능하다.

## NGCF layer와 학습

`NGCFLayer`는 이웃 aggregation, `W1` 변환, element-wise interaction과 `W2` 변환, activation/dropout의 조합이다. 학습 batch는 positive edge와 negative item을 만들고 BPR 계열 pairwise loss 또는 구현된 objective로 사용자 선호 순서를 맞춘다. full-graph forward와 sampled edge loss를 혼동하지 않는다.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train(); loss.backward(); optimizer.step()
model.eval(); with torch.no_grad(): user_features, item_features = model(train_edge_index)
```

## 평가 지표

- **Precision@k** = 추천 Top-k 중 positive item 비율
- **Recall@k** = positive item 중 Top-k에 포함된 비율
- **NDCG@k** = 높은 순위의 hit에 더 큰 가중치를 주는 ranking metric
- **RMSE/MAE** = NCF처럼 실제 rating 값을 예측하는 회귀 지표

NGCF test에서는 train edge로 embedding을 계산하고 test positive edge로 ranking을 평가한다. test edge를 message passing에 넣으면 leakage다. user별 `user_pos_items`를 만든 뒤 candidate item score를 정렬해야 한다.

## 시험 직전 체크

- NCF(embedding+MLP 회귀)와 NGCF(graph propagation+Top-k ranking)를 구분한다.
- explicit rating을 implicit edge로 바꿀 때 threshold의 의미를 설명한다.
- `edge_index` split은 edge 개수 축(`E`) 기준이다.
- `k=10`은 embedding 차원이 아니라 추천 목록 길이다.
- cold-start, 인기 편향, negative sampling, train/test leakage를 한계로 기록한다.
