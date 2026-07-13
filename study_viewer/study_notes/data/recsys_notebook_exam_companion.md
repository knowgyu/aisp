# Recommender Notebook 시험 동반 노트: NGCF·GCF·NCF

원본: `data/7_8-recsys-practice/RecSys_GCF_practice.ipynb`, `RecSys_GCF_sol.ipynb`, `RecSys_NCF.ipynb`.

## 1. 데이터와 graph

평점 dataframe을 `rating_threshold`로 implicit positive interaction으로 바꾼다. 그래프는 user node와 item node, edge는 관측 interaction이다.

```text
edge_index: [2, E]
edge_index[0] = user id
edge_index[1] = num_users + item id  # node id 충돌 방지 offset
user embedding: [num_users, D]
item embedding: [num_items, D]
```

`train_test_split`은 edge 열(`E`) 인덱스를 train/validation/test로 나눈다. adjacency matrix 대신 sparse `edge_index`를 GNN에 넣는 것이 실습의 핵심이다.

## 2. NGCF/GCF data flow

```text
initial user/item embedding
 -> graph message passing (이웃 aggregate)
 -> Linear(W1), interaction transform(W2), activation/dropout
 -> layer별 embedding concat 또는 최종 embedding
 -> score(u,i) = user_embedding · item_embedding
 -> top-k ranking
```

`NGCFLayer(input_dim, output_dim)`의 `W1`, `W2`는 feature transform을 담당한다. 모델 생성 예시는 `embedding_dim=64`, `layer_dims=[64,64]`, `dropout=0.1`이다. 최종 user/item feature는 각각 `[U,D]`, `[I,D]`; 전체 score matrix를 만들면 `[U,I]`가 되므로 큰 데이터에서는 candidate batch로 계산한다.

## 3. 평가 함수

`evaluate(user_features, item_features, test_edge_index, k)`는 각 user의 positive test item을 모은 뒤 점수 상위 k를 고른다. 이미 train에서 본 item은 candidate에서 제외해야 한다.

- `Recall@K = 추천된 relevant item 수 / relevant item 전체 수`
- `Precision@K = 추천된 relevant item 수 / K`
- `NDCG@K`는 순위가 높은 relevant item에 더 큰 gain을 준다.

평가에서 `edge_index[1] - num_users`로 item index를 복원하는 offset을 잊으면 전혀 다른 item이 된다.

## 4. NCF와 비교

| 모델 | 상호작용 함수 | 특징 |
|---|---|---|
| GCF/NGCF | embedding dot product + graph propagation | 협업 구조와 이웃 정보 활용 |
| NCF | user/item embedding을 concat→MLP | 비선형 user-item interaction 학습 |

NCF의 입력은 user id와 item id이며 embedding lookup 결과 `[B,D]` 두 개를 concat해 `[B,2D]`로 만든 뒤 MLP를 통과시킨다. 출력은 보통 `[B,1]` relevance/logit이다.

## 5. 실습 빈칸과 누수 체크

1. positive threshold와 negative sampling 규칙을 먼저 확정한다.
2. train edge만 사용해 message passing/학습하고 test edge는 평가에만 둔다.
3. model은 `train()`에서 dropout을 적용하고, 평가에서는 `eval()`과 `no_grad()`를 쓴다.
4. 평점 예측과 top-k ranking을 구분한다. RMSE만으로 추천 순위를 설명할 수 없다.
5. 인기 item만 추천하는지, user별 candidate 수가 충분한지 확인한다.
