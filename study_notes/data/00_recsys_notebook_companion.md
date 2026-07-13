# Recommender System Notebook Companion: GCF·NCF와 ranking 지표

원본: `data/7_8-recsys-practice/RecSys_GCF_practice.ipynb`, `RecSys_GCF_sol.ipynb`, `RecSys_NCF.ipynb`. 목표는 user/item index가 embedding과 score로 흐르는 과정을 복원하는 것이다.

## 1. 공통 데이터 흐름

```text
(user_id, item_id, label)
 → integer remap
 → user embedding U [B,D], item embedding V [B,D]
 → interaction (dot/product 또는 MLP) [B,D or 2D]
 → score/logit [B]
 → BCE/BPR loss
```

`Embedding(num_users,D)`와 `Embedding(num_items,D)`의 입력은 정수 id `[B]`이고 출력은 `[B,D]`다. positive interaction과 negative sample의 label은 각각 1/0이다.

## 2. GCF와 NCF

| 모델 | 결합 방식 | 핵심 객체 | 장단점 |
|---|---|---|---|
| MF/GCF | `u * v` 또는 graph propagation 후 내적 | user/item embedding, score head | 단순·빠름, 비선형 선호 표현 제한 |
| NCF | GMF 경로와 MLP 경로를 concat | `torch.cat([gmf, mlp], dim=-1)`, `Linear` | 복잡한 상호작용, 파라미터·과적합 증가 |

예측 logits가 `[B,1]`이면 loss 전 `view(-1)` 또는 target을 `[B,1]`로 맞춘다. `BCEWithLogitsLoss`는 sigmoid 전 logits를 받는다.

## 3. 학습·평가 API 순서

```python
model.train()
for user, item, label in loader:
    optimizer.zero_grad()
    logits = model(user, item)
    loss = criterion(logits, label.float())
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    scores = model(all_users, candidate_items)
```

평가에서는 이미 본 item을 후보에서 제외하고, user마다 candidate score를 정렬해 top-k를 만든다. 단순 accuracy는 negative가 많은 추천 문제를 과대평가하므로 ranking metric을 함께 쓴다.

- `HR@K`: 정답 item이 top-k에 있으면 1; 높을수록 좋다.
- `Recall@K`: relevant item 중 top-k에 포함된 비율.
- `NDCG@K`: 순위가 앞일수록 큰 discount; 높을수록 좋다.
- RMSE/MAE: explicit rating 회귀에서 사용; 낮을수록 좋다.

## 4. 시험 함정

- user/item id를 임의의 실수 feature로 넣지 말고 embedding index로 매핑한다.
- train interaction으로 만든 negative가 실제 positive를 침범하지 않도록 sampling한다.
- test에서 학습 중 본 interaction을 정답으로 재평가하면 leakage가 된다.
- GCF의 graph convolution은 단순히 `Linear`를 쌓는 것과 다르다. 이웃 전파/정규화와 embedding 결합을 설명한다.
- ranking 평가는 user별 후보 집합과 cutoff `K`를 고정해야 비교가 가능하다.

## 5. Graph 기반 실습 shape

NGCF/GCF 계열은 user-item bipartite graph를 `edge_index=[2,E]`로 표현한다. 첫 행은 user node index, 둘째 행은 item node index이며, message passing 후 전체 node embedding `[num_users+num_items,D]`에서 user/item 행을 꺼내 score를 계산한다. NCF는 graph 없이 GMF와 MLP 경로를 결합할 수 있으므로 두 실습을 구분한다.
