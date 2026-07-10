# Language / LLM 코드 실습 정답·해설지

이 문서는 생성된 시험 대비 실습 노트북의 `## 정답 입력` 셀에 대응한다.
정답을 바로 복사하기보다 먼저 입력·실행하고, 실패 원인을 기록한 뒤 비교한다.

## 출제 포인트 기준

- embedding, attention, GPT layer의 구성과 tensor 흐름
- next-token/classification/instruction 학습의 loss 연결
- tokenizer와 dataset이 모델 입력을 만드는 과정

## LLM Practice 01. Vector Space 코드 학습

원본: `llm_hands_on/Chapter_1_Exercise_Vector Space.ipynb`
실습본: `practice_notebooks/language/01-vector-space.ipynb`

### Drill 1 — Gensim 실습 예제

원본 Cell `002`. 이 셀은 **Gensim 실습 예제** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
import gensim.downloader as api

def run_gensim_tutorial():
    # 1. 사전 학습된 가벼운 모델 다운로드 및 로드 (약 65MB)
    # 강의 중 첫 실행 시 다운로드 시간이 약간 소요될 수 있습니다.
    print("모델을 로딩 중입니다... (glove-wiki-gigaword-50)\n")
    model = api.load("glove-wiki-gigaword-50")

    # ----------------------------------------------------
    # [기능 1] 단어의 의미(유사도) 계산하기
    # 두 단어 벡터 간의 코사인 유사도(Cosine Similarity)를 계산합니다.
    # ----------------------------------------------------
    print("1. 단어 간 유사도 계산하기")
    word1, word2, word3 = 'cat', 'dog', 'car'

    sim_cat_dog = model.similarity(word1, word2)
    sim_cat_car = model.similarity(word1, word3)

    print(f" - '{word1}'와 '{word2}'의 유사도: {sim_cat_dog:.4f}")
    print(f" - '{word1}'와 '{word3}'의 유사도: {sim_cat_car:.4f}\n")

    # ----------------------------------------------------
    # [기능 2] 유사한 단어 가져오기
    # 특정 단어와 벡터 공간상에서 가장 가까운 단어들을 추출합니다.
    # ----------------------------------------------------
    print("2. 'computer'와 가장 유사한 단어 5개 가져오기")
    target_word = 'computer'
    similar_words = model.most_similar(target_word, topn=5)

    for word, score in similar_words:
        print(f" - {word} (유사도 점수: {score:.4f})")
    print()

    # ----------------------------------------------------
    # [기능 3] 주어진 단어들을 그룹으로 분류 (이질적인 단어 찾기)
    # 단어들의 의미적 군집을 파악하여, 그룹에 어울리지 않는 단어를 찾아냅니다.
    # ----------------------------------------------------
    print("3. 단어 그룹 중 성격이 다른 단어(Outlier) 분류하기")
    word_group = ['apple', 'banana', 'orange', 'car']

    outlier = model.doesnt_match(word_group)
    print(f" - 단어 그룹: {word_group}")
    print(f" - 과일 그룹에 어울리지 않는 단어: '{outlier}'\n")

# 실행
if __name__ == "__main__":
    run_gensim_tutorial()
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

## LLM Practice 02. Dataset / Tokenizer 코드 학습

원본: `llm_hands_on/Chapter_2_Exercise_Dataset.ipynb`
실습본: `practice_notebooks/language/02-dataset.ipynb`

### Drill 1 — 핵심 코드 골격

원본 Cell `000`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
import importlib
import tiktoken

print("tiktoken version:", importlib.metadata.version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
# region  [token encoding]
# TODO: 인코딩 메서드명을 채워 text를 토큰 ID로 변환하세요.
# 힌트: `????`에는 문자열을 토큰 ID 리스트로 바꾸는 인코딩 메서드 이름이 들어갑니다.
# 힌트: `tokenizer.` 뒤 메서드 이름만 채우고 `allowed_special` 설정은 그대로 두면 됩니다.
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# endregion
print(integers)

# region  [token decoding]
# TODO: 디코딩 메서드명을 채워 토큰 ID를 문자열로 복원하세요.
# 힌트: `????`에는 토큰 ID 리스트를 다시 문자열로 바꾸는 디코딩 메서드 이름이 들어갑니다.
# 힌트: `tokenizer.` 뒤 메서드 이름만 채우고 입력은 `integers` 그대로 사용하세요.
strings = tokenizer.decode(integers)
# endregion
print(strings)
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — 핵심 코드 골격

원본 Cell `002`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### 중요: 이 코드는 GPT 모델 학습을 위한 데이터셋 클래스를 정의합니다

import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        """
        Args:
            txt (str): 학습할 전체 텍스트 데이터
            tokenizer: 텍스트를 토큰 ID로 변환해주는 토크나이저 (예: tiktoken)
            max_length (int): 모델이 한 번에 볼 수 있는 윈도우 크기 (입력 시퀀스 길이)
            stride (int): 윈도우를 이동시킬 간격 (데이터 중복 정도를 결정)
        """
        self.input_ids = []
        self.target_ids = []

        # 1. 전체 텍스트 토큰화
        # 텍스트를 정수 리스트(token_ids)로 변환합니다.
        # <|endoftext|> 같은 특수 토큰도 허용하여 인코딩합니다.
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 데이터가 너무 짧으면 학습할 수 없으므로 최소 길이를 확인합니다.
        assert len(token_ids) > max_length, "토큰화된 입력의 개수는 적어도 max_length+1과 같아야 합니다."

        # 2. 슬라이딩 윈도우(Sliding Window)로 데이터 생성
        # 전체 토큰 리스트를 훑으며 max_length 길이만큼 잘라냅니다.
        # stride만큼 건너뛰며 반복합니다.
        for i in range(0, len(token_ids) - max_length, stride):
            # 입력 청크: 현재 위치(i)부터 max_length만큼 가져옵니다.
            # region [input 청크 생성]
            # TODO: 입력 청크 슬라이싱 구간을 채우세요.
            # 힌트: 첫 번째 `????`는 슬라이싱 시작 위치인 `i`입니다.
            # 힌트: 두 번째 `????`도 끝 위치 계산에 쓰이는 `i`라서 `i : i + max_length` 형태를 완성하면 됩니다.
            input_chunk = token_ids[i : i + max_length]
            # endregion

            # 타겟 청크: 입력보다 1칸 뒤의 위치(i+1)부터 가져옵니다.
            # GPT는 '다음 단어'를 맞추는 모델이므로, 정답은 입력보다 한 칸씩 뒤로 밀려있어야 합니다.
            # region [target 청크 생성]
            # TODO: 타깃 청크 슬라이싱 구간을 채우세요.
            # 힌트: 네 개의 `????`는 순서대로 `i`, `1`, `i`, `1`입니다.
            # 힌트: 입력보다 한 칸 뒤를 가리키도록 `i + 1 : i + max_length + 1` 형태를 만들면 됩니다.
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            # endregion

            # 추출한 데이터를 텐서(Tensor)로 변환하여 리스트에 저장합니다.
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # 데이터셋의 총 샘플(청크) 개수를 반환합니다.
        return len(self.input_ids)

    def __getitem__(self, idx):
        # DataLoader가 데이터를 요청할 때 호출됩니다.
        # 해당 인덱스(idx)의 입력과 정답 쌍을 반환합니다.
        # TODO: __getitem__에서 같은 idx의 입력/타깃 쌍을 반환하세요.
        # 힌트: `self.input_ids`와 `self.target_ids`에서 `idx` 위치의 값을 각각 반환하세요.
        return self.input_ids[idx], self.target_ids[idx]
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 3 — 핵심 코드 골격

원본 Cell `003`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
##중요 : 아래 함수는 DataLoader를 생성하는 헬퍼 함수입니다.
import tiktoken
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # 토크나이저를 초기화합니다.
    tokenizer = tiktoken.get_encoding("gpt2")

    # 데이터셋을 만듭니다.
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 데이터 로더를 만듭니다.
    # region [DataLoader 생성]
    # TODO: DataLoader의 첫 번째 인자를 채우세요.
    # 힌트: `????`에는 바로 위에서 생성한 `dataset` 객체가 들어갑니다.
    # 힌트: DataLoader의 첫 번째 위치 인자는 순회할 데이터셋 자체입니다.
    dataloader = DataLoader(
        ????,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    # endregion

    return dataloader
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 4 — 핵심 코드 골격

원본 Cell `006`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

context_length = max_length
# region [포지셔널 임베딩 레이어 생성]
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
# endregion
# region [포지셔널 임베딩 적용]
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
# endregion
print(pos_embeddings.shape)
# region [토큰 임베딩과 포지셔널 임베딩 합산]
input_embeddings = token_embeddings + pos_embeddings
# endregion
print(input_embeddings.shape)
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

## LLM Practice 03. Attention 코드 학습

원본: `llm_hands_on/Chapter_3_Excercise_Attention.ipynb`
실습본: `practice_notebooks/language/03-attention.ipynb`

### Drill 1 — 핵심 코드 골격

원본 Cell `000`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### 중요: 이 코드는 인과적 어텐션 메커니즘을 구현한 PyTorch 모듈입니다.
import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        """
        Args:
            d_in: 입력 벡터의 차원 크기
            d_out: 출력(및 쿼리/키/밸류) 벡터의 차원 크기
            context_length: 모델이 한 번에 처리할 수 있는 최대 문맥 길이 (토큰 수)
            dropout: 드롭아웃 확률
            qkv_bias: 선형 레이어에 편향(bias)을 사용할지 여부
        """
        super().__init__()

        # 1. 쿼리(Query), 키(Key), 밸류(Value)를 만들기 위한 선형 투영 레이어 정의
        # 입력 벡터(d_in)를 각각의 목적에 맞는 벡터(d_out)로 변환합니다.
        # TODO: Q/K/V 투영 레이어 타입과 인자를 채우세요.
        # 힌트: 입력 d_in, 출력 d_out을 사용하는 Linear 레이어를 선언하세요.
        self.W_query = nn.????(????, ????, bias=qkv_bias)
        self.W_key   = nn.????(????, ????, bias=qkv_bias)
        self.W_value = nn.????(????, ????, bias=qkv_bias)

        # 2. 과적합(Overfitting) 방지를 위한 드롭아웃 설정
        self.dropout = nn.Dropout(dropout)

        # 3. 인과적 마스크(Causal Mask) 생성 및 버퍼 등록
        # 'register_buffer'를 사용하면 역전파(학습) 대상은 아니지만, 모델의 상태(state_dict)로 저장됩니다.
        # torch.triu(..., diagonal=1): 대각선(0) 위쪽 삼각형 부분만 1로 채웁니다.
        # 즉, '미래의 정보' 위치에 1을 표시하여 나중에 가릴(masking) 준비를 합니다.
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        # x.shape: [배치 크기(b), 토큰 개수(num_tokens), 입력 차원(d_in)]
        b, num_tokens, d_in = x.shape

        # 입력 x를 통과시켜 현재 시점의 관심사(Query), 검색 대상(Key), 정보 내용(Value)을 추출합니다.
        # region [Q, K, V 벡터 계산]
        # TODO: Q, K, V 계산 호출 대상을 채우세요.
        # 힌트: W_query, W_key, W_value를 각각 x에 적용하면 됩니다.
        keys = self.????(????)        # Shape: [b, num_tokens, d_out]
        queries = self.?????(????)   # Shape: [b, num_tokens, d_out]
        values = self.?????(????)    # Shape: [b, num_tokens, d_out]
        # endregion

        # Query와 Key의 내적(Dot Product)을 통해 각 토큰 간의 관련성을 구합니다.
        # keys.transpose(1, 2): 행렬 곱을 위해 차원을 뒤집습니다. (d_out 차원끼리 곱해짐)
        # region [어텐션 스코어(유사도) 계산]
        # TODO: 어텐션 스코어 행렬곱 피연산자를 채우세요.
        # 힌트: queries와 keys.transpose(1, 2)를 곱해 유사도 점수를 구합니다.
        attn_scores = ???? @ ????.transpose(1, 2)
        #endregion

        # mask가 1인 위치(미래 시점의 토큰들)를 -무한대(-inf)로 채웁니다.
        # 이렇게 하면 나중에 Softmax를 거칠 때 확률이 0이 되어, 미래 정보를 참조하지 못하게 됩니다.
        # [:num_tokens, :num_tokens]: 입력 길이가 context_length보다 짧을 때를 대비해 크기를 맞춥니다.
        # region [인과적 마스킹 (Masking) - 미래 정보 차단]
        # TODO: 미래 시점의 어텐션 스코어를 가릴 값을 채우세요.
        # 힌트: Softmax를 통과하면 0이 되도록 마이너스 무한대(-torch.inf)를 입력하세요.
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens],
            ????
        )
        # endregion

        # 스케일링(/ keys.shape[-1]**0.5): 차원이 커질수록 내적 값이 커져 기울기 소실이 오는 것을 방지합니다.
        # Softmax: 점수를 확률(0~1 사이, 합은 1)로 변환합니다.
        # region [어텐션 가중치(Weights) 계산 및 스케일링]
        # TODO: 어텐션 가중치 함수명을 채우세요.
        # 힌트: 스케일 적용 후 softmax를 사용해 확률로 변환하세요.
        attn_weights = torch.????(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        # endregion

        # 계산된 가중치 중 일부를 무작위로 0으로 만들어 모델이 특정 토큰에만 의존하는 것을 막습니다.
        # region [드롭아웃 적용]
        attn_weights = self.dropout(attn_weights)
        # endregion

        # 어텐션 가중치(확률)를 기반으로 Value(정보)들을 가중 합산합니다.
        # 결과적으로 "현재 토큰과 관련이 깊은 과거 토큰들의 정보"가 진하게 섞인 벡터가 됩니다.
        # region [문맥 벡터(Context Vector) 생성]
        # TODO: 컨텍스트 벡터 계산 피연산자를 채우세요.
        # 힌트: attn_weights와 values를 곱해 최종 컨텍스트 벡터를 얻습니다.
        context_vec = attn_weights @ ????
        # endregion

        return context_vec
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — 핵심 코드 골격

원본 Cell `001`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

d_in = inputs.shape[1]   # 입력 차원 (d=3)
d_out = 2                # Q, K, V의 출력 차원 (d=2)

batch = torch.stack((inputs, inputs), dim=0)
# --- 실행 예시 ---
torch.manual_seed(123)

# 가정: batch 변수가 이미 정의되어 있다고 가정 (예: b=2, num_tokens=6, d_in=...)
# context_length는 모델이 허용하는 최대 길이이므로, 현재 배치의 길이와 같거나 더 길게 설정합니다.
context_length = batch.shape[1]

print("context_length:", context_length)

# TODO: 어텐션 모듈 호출 대상을 채우세요.
# 힌트: 직전에 생성한 CausalAttention 인스턴스(ca)를 호출하면 됩니다.
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ????(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 3 — 핵심 코드 골격

원본 Cell `002`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


torch.manual_seed(123)

context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 4 — 핵심 코드 골격

원본 Cell `003`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # As in `CausalAttention`, for inputs where `num_tokens` exceeds `context_length`,
        # this will result in errors in the mask creation further below.
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs
        # do not exceed `context_length` before reaching this forward method.

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec

torch.manual_seed(123)

batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)

context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

## LLM Practice 03+. Attention Visualization 코드 학습

원본: `llm_hands_on/Chapter_3_Excercise_Viz_Multi_head_attention.ipynb`
실습본: `practice_notebooks/language/03-attention-visualization.ipynb`

### Drill 1 — Multi-Head Attention Visualized

원본 Cell `001`. 이 셀은 **Multi-Head Attention Visualized** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
import sys
sys.path.append('.')
import torch
import tiktoken
import matplotlib.pyplot as plt
from previous_chapters import load_gpt2_model

# GPT-2 Small (124M) 설정
GPT_CONFIG_SMALL = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": True
}

sentence_a = "The artist painted the portrait of a woman with a brush"

# 모델 로드
model = load_gpt2_model("gpt2-small-124M.pth", GPT_CONFIG_SMALL)
model.eval()

# 토크나이저
tokenizer = tiktoken.get_encoding("gpt2")

# 토큰화 및 attention weights 수집 (forward hook)
token_ids = tokenizer.encode(sentence_a)
tokens = [tokenizer.decode([t]) for t in token_ids]
input_tensor = torch.tensor([token_ids])

all_attentions = []

def make_hook():
    def hook_fn(module, input, output):
        x = input[0]
        b, num_tokens, _ = x.shape
        keys = module.W_key(x)
        queries = module.W_query(x)
        keys = keys.view(b, num_tokens, module.num_heads, module.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, module.num_heads, module.head_dim).transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = module.mask.bool()[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        all_attentions.append(attn_weights.detach())
    return hook_fn

hooks = [block.att.register_forward_hook(make_hook()) for block in model.trf_blocks]
with torch.no_grad():
    model(input_tensor)
for hook in hooks:
    hook.remove()

print(f"레이어 수: {len(all_attentions)}, 헤드 수: {all_attentions[0].shape[1]}, 토큰: {tokens}")
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — Multi-Head Attention Visualized

원본 Cell `002`. 이 셀은 **Multi-Head Attention Visualized** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
# head_view 대체: layer=4의 모든 헤드 attention 히트맵
layer_idx = 4
num_heads = all_attentions[layer_idx].shape[1]  # 16

fig, axes = plt.subplots(2, num_heads // 2, figsize=(24, 8))
axes = axes.flatten()

for head_idx in range(num_heads):
    attn = all_attentions[layer_idx][0, head_idx].numpy()
    axes[head_idx].imshow(attn, cmap="Blues")
    axes[head_idx].set_title(f"Head {head_idx}", fontsize=9)
    axes[head_idx].set_xticks(range(len(tokens)))
    axes[head_idx].set_xticklabels(tokens, rotation=90, fontsize=7)
    axes[head_idx].set_yticks(range(len(tokens)))
    axes[head_idx].set_yticklabels(tokens, fontsize=7)

plt.suptitle(f"Head View - Layer {layer_idx} (GPT-2 Medium)", fontsize=13)
plt.tight_layout()
plt.show()
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 3 — Multi-Head Attention Visualized

원본 Cell `003`. 이 셀은 **Multi-Head Attention Visualized** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
import io
import numpy as np
import ipywidgets as widgets
from IPython.display import display

previous_ui = globals().get("_attention_ui")
if previous_ui is not None:
    previous_ui["controls"].close()
    previous_ui["layer_w"].close()
    previous_ui["head_w"].close()
    previous_ui["plot_widget"].close()

plot_widget = widgets.Image(format="png")

def plot_attention_lines(change=None):
    layer_idx = layer_w.value
    head_idx = head_w.value
    attn = all_attentions[layer_idx][0, head_idx].numpy()
    n = len(tokens)

    with plt.ioff():
        fig, ax = plt.subplots(figsize=(8, max(6, n * 0.45)))

    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")

    y_pos = np.linspace(0.95, 0.05, n)
    x_left, x_right = 0.2, 0.8

    for i in range(n):
        for j in range(n):
            w = float(attn[i, j])
            if w > 0.005:
                ax.plot([x_left, x_right], [y_pos[i], y_pos[j]],
                        color="#4A90D9", alpha=min(w * 2, 1.0), linewidth=1.2)

    for i, tok in enumerate(tokens):
        ax.text(x_left - 0.01, y_pos[i], tok, ha="right", va="center",
                color="white", fontsize=10, fontfamily="monospace")
        ax.text(x_right + 0.01, y_pos[i], tok, ha="left", va="center",
                color="white", fontsize=10, fontfamily="monospace")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(f"Layer: {layer_idx}  Head: {head_idx}",
                 color="white", fontsize=12, pad=10)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plot_widget.value = buffer.getvalue()
    buffer.close()
    plt.close(fig)

layer_w = widgets.Dropdown(
    options=list(range(len(all_attentions))), value=4, description="Layer:")
head_w = widgets.Dropdown(
    options=list(range(all_attentions[0].shape[1])), value=3, description="Head:")
controls = widgets.HBox([layer_w, head_w])

layer_w.observe(plot_attention_lines, names="value")
head_w.observe(plot_attention_lines, names="value")

display(controls)
display(plot_widget)
globals()["_attention_ui"] = {
    "layer_w": layer_w,
    "head_w": head_w,
    "controls": controls,
    "plot_widget": plot_widget,
}
plot_attention_lines()  # ?? ???
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

## LLM Practice 04. GPT Architecture 코드 학습

원본: `llm_hands_on/Chapter_4_Excercise_GPT.ipynb`
실습본: `practice_notebooks/language/04-gpt.ipynb`

### Drill 1 — 핵심 코드 골격

원본 Cell `000`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#####################################
# Chapter 2: 데이터 로딩 및 처리
#####################################

class GPTDatasetV1(Dataset):
    """
    GPT 학습을 위한 데이터셋 클래스입니다.
    텍스트를 입력받아 토큰화하고, 입력(input)과 타겟(target) 쌍을 만듭니다.
    """
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 1. 전체 텍스트를 토큰화합니다.
        # <|endoftext|> 같은 특수 토큰도 허용하여 인코딩합니다.
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 2. 슬라이딩 윈도우 방식으로 데이터를 조각냅니다.
        # stride만큼 이동하면서 max_length 길이의 덩어리(chunk)를 만듭니다.
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            # 타겟은 입력보다 한 칸 뒤의 토큰들입니다 (다음 토큰 예측 과제).
            target_chunk = token_ids[i + 1: i + max_length + 1]

            # 텐서로 변환하여 저장
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # 데이터셋의 총 샘플 수 반환
        return len(self.input_ids)

    def __getitem__(self, idx):
        # 특정 인덱스의 입력과 타겟 쌍 반환
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """
    텍스트 데이터를 받아 학습에 사용할 DataLoader를 생성하는 함수입니다.
    """
    # 토크나이저 초기화 (GPT-2용 BPE 인코딩 사용)
    tokenizer = tiktoken.get_encoding("gpt2")

    # 데이터셋 생성
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 데이터로더 생성 (배치 단위로 데이터를 묶어주고 셔플링 수행)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — 핵심 코드 골격

원본 Cell `001`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python

#####################################
# Chapter 3: 어텐션 메커니즘 (모델의 핵심)
#####################################
class MultiHeadAttention(nn.Module):
    """
    멀티 헤드 셀프 어텐션 (Multi-Head Self-Attention) 모듈입니다.
    입력 데이터 간의 관계성을 여러 관점(Head)에서 병렬로 학습합니다.
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "출력 차원(d_out)은 헤드 수(num_heads)로 나누어 떨어져야 합니다."

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 각 헤드가 담당할 차원 크기

        # Query, Key, Value를 만들기 위한 선형 투영 레이어들
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 멀티 헤드 결과를 하나로 합친 후 통과시키는 출력 레이어
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Causal Mask (인과적 마스킹) 생성: 미래의 토큰을 보지 못하게 함
        # 상삼각 행렬(대각선 위쪽)을 1로 채워서 나중에 마스킹에 사용
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape # b: 배치 크기, num_tokens: 시퀀스 길이

        # 1. Q, K, V 계산
        keys = self.W_key(x)     # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 2. 헤드 나누기 (Multi-head splitting)
        # 차원을 변형하여 여러 헤드가 병렬로 처리할 수 있게 함
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 3. 차원 순서 변경 (Transpose)
        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # 이렇게 하면 (num_tokens, head_dim) 행렬이 헤드 개수만큼 독립적으로 존재하게 됨
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 4. Scaled Dot-Product Attention 계산
        # Query와 Key의 내적 (유사도 계산)
        attn_scores = queries @ keys.transpose(2, 3)  # 결과 Shape: (b, num_heads, num_tokens, num_tokens)

        # 5. 마스킹 (Masking)
        # 현재 시점보다 미래의 토큰 정보를 참조하지 못하게 -inf로 가림
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 6. 소프트맥스 및 드롭아웃
        # 점수를 확률로 변환 (합이 1이 되도록)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 7. Value와의 가중치 합 (Context Vector 계산)
        # Shape: (b, num_heads, num_tokens, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 8. 헤드 결합 (Concatenation)
        # 나눠졌던 헤드들을 다시 원래의 d_out 차원으로 합침
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # 9. 최종 선형 투영
        context_vec = self.out_proj(context_vec)

        return context_vec
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 3 — 핵심 코드 골격

원본 Cell `002`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python

#####################################
# Chapter 4: GPT 아키텍처 구성 요소
#####################################
class LayerNorm(nn.Module):
    """
    층 정규화 (Layer Normalization): 학습 안정성을 높임
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim)) # 학습 가능한 스케일 파라미터 (Gamma)
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # 학습 가능한 시프트 파라미터 (Beta)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # TODO: LayerNorm 계산식의 통계량 변수를 채우세요.
        # 힌트: 평균(mean)과 분산(var)을 사용해 정규화합니다.
        norm_x = (x - ????) / torch.sqrt(???? + self.eps)
        # TODO: LayerNorm 학습 파라미터를 채우세요.
        # 힌트: scale로 곱하고 shift를 더하는 형태를 사용합니다.
        return self.???? * norm_x + self.????


class GELU(nn.Module):
    """
    GELU 활성화 함수: GPT 계열에서 주로 사용하는 비선형 함수
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    피드 포워드 네트워크 (Feed-Forward Network)
    어텐션이 모은 정보를 각 토큰별로 개별적으로 가공하는 역할
    보통 임베딩 차원을 4배로 늘렸다가 다시 줄임
    """
    def __init__(self, cfg):
        super().__init__()
        # TODO: FeedForward의 확장 배수와 활성화 함수 클래스를 채우세요.
        # 힌트: GPT 구조에서는 hidden 크기를 4배로 확장합니다. 확장 Linear와 축소 Linear 사이에 GELU 활성화 함수를 넣으세요.
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], ???? * cfg["emb_dim"]),
            ????(),
            nn.Linear(???? * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 4 — 핵심 코드 골격

원본 Cell `003`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
##중요 : 아래는 GPT 모델 전체 구조를 정의하는 코드입니다

#####################################
# Chapter 4: GPT 아키텍처 조립
#####################################
class TransformerBlock(nn.Module):
    """
    표준 트랜스포머 블록 (Decoder Block)
    구조: LayerNorm -> Attention -> Add(Residual) -> LayerNorm -> FeedForward -> Add(Residual)
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # region [어텐션 블록 (Residual Connection 적용)]
        shortcut = x
        x = self.norm1(x) # Pre-LayerNorm 방식
        # TODO
        x = self.????(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 원본 입력을 더해줌 (기울기 소실 방지)
        # endregion

        # region [피드 포워드 블록 (Residual Connection 적용)]
        shortcut = x
        x = self.norm2(x)
        # TODO
        x = self.????(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 원본 입력 다시 더함
        # endregion

        return x
class GPTModel(nn.Module):
    """
    전체 GPT 모델 구조 정의
    Embedding -> Transformer Blocks -> Final Norm -> Output Head
    """
    def __init__(self, cfg):
        super().__init__()
        # 토큰 임베딩 (단어 -> 벡터)
        # TODO
        self.tok_emb = nn.Embedding(cfg["????"], cfg["emb_dim"])
        # 위치 임베딩 (위치 정보 -> 벡터)
        # TODO
        self.pos_emb = nn.Embedding(cfg["????"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 트랜스포머 블록 쌓기 (n_layers 만큼)
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # 최종 정규화 및 출력 헤드
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # TODO
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["????"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        # region [임베딩 생성]
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # endregion

        # region [토큰 임베딩과 위치 임베딩 합산]
        # TODOe
        x = tok_embeds + ????
        # endregion

        x = self.drop_emb(x)

        # region [트랜스포머 블록 통과]
        # TODO
        x = self.????(x)
        # endregion

        # region [최종 출력 계산]
        x = self.final_norm(x)
        # TODO
        logits = self.????(x) # 각 단어에 대한 예측 점수 (Logits)
        # endregion

        return logits
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

## LLM Practice 05. Pretraining 코드 학습

원본: `llm_hands_on/Chapter_5_Excercise_Pretraining.ipynb`
실습본: `practice_notebooks/language/05-pretraining.ipynb`

### Drill 1 — 핵심 코드 골격

원본 Cell `001`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### 중요: lOSS를 계산합니다.
def calc_loss_batch(input_batch, target_batch, model, device):
    """
    하나의 배치(Batch)에 대해 모델의 예측값과 실제값 사이의 오차(Loss)를 계산합니다.
    """
    # 데이터를 GPU(또는 설정된 device)로 이동
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    # 1. 모델의 순전파(Forward Pass)
    # logits shape: (batch_size, sequence_length, vocab_size)
    logits = model(input_batch)

    # 2. 손실 계산 (CrossEntropyLoss)
    # PyTorch의 CrossEntropyLoss는 입력을 (N, C) 형태로 받기를 원합니다.
    #   - N: 전체 샘플 수 (여기서는 Batch_size * Sequence_length)
    #   - C: 클래스 수 (여기서는 Vocab_size)
    # 따라서 3차원 텐서를 2차원으로 평탄화(flatten) 해야 합니다.

    # logits.flatten(0, 1) -> (batch_size * sequence_length, vocab_size)
    # target_batch.flatten() -> (batch_size * sequence_length)
    # region [손실 계산 (CrossEntropyLoss)]
    # TODO: 손실 함수명과 logits/target 평탄화 메서드를 채우세요.
    # 힌트: cross_entropy에 logits.flatten(0, 1)과 target_batch.flatten()을 전달합니다.
    loss = torch.nn.functional.????(
        logits.????(0, 1),
        target_batch.????()
    )
    # endregion
    return loss
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — 핵심 코드 골격

원본 Cell `002`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### 중요: 모델을 학습합니다.
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    """
    [핵심] 모델 훈련을 담당하는 메인 루프입니다.
    """
    # 로그 저장을 위한 리스트들
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # Epoch: 전체 데이터셋을 한 번 훑는 단위
    for epoch in range(num_epochs):
        # region [train 모드 설정]
        # TODO: 모델 모드 전환 메서드를 채우세요.
        # 힌트: 학습 루프 시작 전에는 train 모드를 설정해야 합니다.
        model.????()  # 훈련 시작 전 반드시 train 모드 설정
        # endregion

        for input_batch, target_batch in train_loader:
            # [Step 1] 이전 배치에서 계산된 기울기 초기화 (필수)
            # region [기울기 초기화]
            # TODO: 기울기 초기화 메서드를 채우세요.
            # 힌트: 배치마다 이전 gradient가 누적되지 않도록 zero_grad()를 먼저 호출합니다.
            optimizer.????()
            # endregion
            #
            # [Step 2] 순전파 및 손실 계산
            # region [손실 계산]
            # TODO: 배치 손실 계산 함수명을 채우세요.
            # 힌트: input_batch, target_batch, model, device를 인자로 받는 헬퍼를 호출하면 됩니다.
            loss = ????(input_batch, target_batch, model, device)
            # endregion

            # [Step 3] 역전파 (Backpropagation): 각 파라미터별 기울기(Gradient) 계산
            # region [역전파 수행]
            # TODO: 역전파 메서드를 채우세요.
            # 힌트: loss 텐서에서 backward()를 호출해 gradient를 계산합니다.
            loss.????()
            # endregion

            # [Step 4] 가중치 업데이트: 계산된 기울기를 이용해 파라미터 수정
            # region [가중치 업데이트]
            # TODO: 파라미터 업데이트 메서드를 채우세요.
            # 힌트: backward()로 계산된 gradient를 이용해 optimizer.step()으로 가중치를 갱신합니다.
            optimizer.????()
            # endregion

            tokens_seen += input_batch.numel() # 처리한 토큰 수 카운트
            global_step += 1

            # 일정 스텝마다 검증(Evaluation) 수행 및 로그 출력
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 한 에포크가 끝날 때마다 샘플 문장을 생성하여 모델이 똑똑해지고 있는지 확인
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 3 — 핵심 코드 골격

원본 Cell `003`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
### 중요: 결과를 생성합니다.
def generate(model, idx, max_new_tokens, context_size, eos_id=None):
    """
    고급 텍스트 생성 함수입니다. (Top-k 샘플링 & Temperature Scaling 포함)

    Args:
        idx: 현재 문맥 (Shape: Batch, Time)
        temperature: 0이면 결정적(Greedy), 높을수록 창의적/랜덤
        top_k: 확률 상위 k개만 후보로 남김 (이상한 단어 생성 방지)
    """

    # 지정된 토큰 수만큼 반복 생성
    for _ in range(max_new_tokens):

        # 1. 문맥 자르기 (Context Cropping)
        # 모델이 처리할 수 있는 최대 길이(context_size)를 넘지 않도록 뒤쪽만 남김
        idx_cond = idx[:, -context_size:]

        # 2. 모델 예측
        # region [모델 예측]
        with torch.no_grad():
            logits = model(idx_cond)
        # endregion

        # 3. 다음 단어 예측을 위해 '마지막 시점'의 로짓만 추출
        # logits shape: (batch, seq_len, vocab_size) -> (batch, vocab_size)
        # region [마지막 시점 로짓 추출]
        # TODO: 마지막 시점 선택 인덱스를 채우세요.
        # 힌트: 다음 토큰 생성은 마지막 위치 logits만 사용합니다.
        logits = logits[:, ????, :]
        # endregion

        # 가장 높은 확률을 가진 토큰 선택 (Greedy Decoding)
        # region [다음 토큰 선택 (Greedy Decoding)]
        # TODO: 다음 토큰 선택 함수명을 채우세요.
        # 힌트: Greedy decoding에서는 argmax를 사용합니다.
        idx_next = torch.????(logits, dim=-1, keepdim=True)
        # endregion

        # 4. 종료 조건 확인 (EOS 토큰이 나오면 중단)
        if idx_next == eos_id:
            break

        # 5. 생성된 토큰 이어붙이기
        # 기존 문장(idx) 뒤에 새로 뽑은 토큰(idx_next)을 붙여 다음 스텝의 입력으로 씀
        # region [생성된 토큰 이어붙이기]
        # TODO: concat 대상 변수를 채우세요.
        # 힌트: 방금 고른 idx_next를 기존 idx 뒤에 붙입니다.
        idx = torch.cat((idx, ????), dim=1)
        # endregion

    return idx
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 4 — 핵심 코드 골격

원본 Cell `004`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
def text_to_token_ids(text, tokenizer):
    """
    사람이 읽는 텍스트 문자열을 모델이 이해하는 토큰 ID 텐서로 변환합니다.
    """
    encoded = tokenizer.encode(text)

    # 모델은 입력으로 (Batch_Size, Sequence_Length)의 2차원 텐서를 요구합니다.
    # 하지만 여기서는 문장 1개이므로 Batch_Size가 없습니다.
    # 따라서 .unsqueeze(0)를 사용하여 가짜 배치 차원을 추가합니다.
    # 예: [12, 34, 56] -> [[12, 34, 56]] (Shape: [1, seq_len])
    # TODO: 배치 차원 추가 메서드를 채우세요.
    # 힌트: 단일 시퀀스를 모델 입력 형태로 만들기 위해 unsqueeze(0)를 사용합니다.
    encoded_tensor = torch.tensor(encoded).????(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """
    모델이 뱉어낸 토큰 ID 텐서를 사람이 읽을 수 있는 텍스트로 변환합니다.
    """
    # 디코딩을 위해 불필요한 배치 차원을 제거합니다.
    # 예: [[12, 34, 56]] -> [12, 34, 56]
    # TODO: 배치 차원 제거 메서드를 채우세요.
    # 힌트: 디코딩 전에는 squeeze(0)로 앞쪽 차원을 제거합니다.
    flat = token_ids.????(0)
    return tokenizer.decode(flat.tolist())


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    데이터 로더 전체(또는 일부)를 돌면서 평균 손실을 계산합니다.
    훈련에는 관여하지 않고, 오직 '평가(Evaluation)' 목적으로만 쓰입니다.
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # 평가 시간을 단축하기 위해 전체 데이터를 다 보지 않고 일부만 볼 수 있게 설정
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item() # 텐서에서 실수값(float)만 추출하여 누적
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    현재 모델의 성능을 훈련 세트와 검증 세트 각각에 대해 평가합니다.
    """
    model.eval() # [중요] 평가 모드 전환: Dropout이나 BatchNorm 등의 동작이 변경됨

    with torch.no_grad(): # [중요] 기울기(Gradient) 계산 끔 -> 메모리 절약 및 속도 향상
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    model.train() # 평가가 끝나면 다시 훈련 모드로 복귀해야 함
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    훈련 중간중간 모델이 문장을 어떻게 생성하는지 눈으로 확인하기 위한 함수입니다.
    """
    model.eval() # 평가 모드
    context_size = model.pos_emb.weight.shape[0] # 모델이 처리 가능한 최대 길이
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)

        # 출력이 너무 길어지면 보기 힘드므로 줄바꿈을 공백으로 변경
        print(decoded_text.replace("\n", " "))

    model.train() # 훈련 모드 복귀



def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    훈련 진행 상황(Loss 변화)을 시각화하는 함수입니다.
    X축을 'Epoch'와 '처리한 토큰 수' 두 가지 기준으로 보여줍니다.
    """
    fig, ax1 = plt.subplots()

    # 기본 X축: Epoch 기준
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # 보조 X축 (상단): 처리한 토큰 수(Tokens seen) 기준
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0) # 투명한 그래프로 축만 생성
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    # plt.show() # 주피터 노트북 환경이면 주석 해제


def main(gpt_config, settings):
    # 랜덤 시드 고정 (실험 재현성을 위해)
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##############################
    # 1. 데이터 준비
    ##############################
    file_path = "datas/the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    ##############################
    # 2. 모델 및 옵티마이저 초기화
    ##############################
    model = GPTModel(gpt_config)
    model.to(device) # 모델을 GPU 메모리로 올림

    # AdamW: 가중치 감쇠(Weight Decay)가 적용된 Adam 옵티마이저
    # Weight Decay는 모델이 너무 복잡해지지 않도록 규제(Regularization)하는 역할
    # TODO: AdamW에 넘길 파라미터 이터레이터와 학습률 키워드를 채우세요.
    # 힌트: 첫 번째 인자는 model.parameters(), 학습률은 lr=settings["learning_rate"] 형태입니다.
    optimizer = torch.optim.AdamW(
        model.????(), ????=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    ##############################
    # 3. 데이터 로더 구축
    ##############################
    # 전체 텍스트를 9:1 비율로 훈련용과 검증용으로 나눔
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    # 훈련 데이터 로더: 순서를 섞음(Shuffle=True)
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    # 검증 데이터 로더: 순서를 섞지 않음(Shuffle=False) -> 평가는 일관되게
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    ##############################
    # 4. 훈련 시작
    ##############################
    tokenizer = tiktoken.get_encoding("gpt2") # BPE 토크나이저

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    return train_losses, val_losses, tokens_seen, model
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

## LLM Practice 06. Classification Fine-tuning 코드 학습

원본: `llm_hands_on/Chapter_6_Excercise_Finetuning_Classification.ipynb`
실습본: `practice_notebooks/language/06-classification-finetuning.ipynb`

### Drill 1 — 핵심 코드 골격

원본 Cell `000`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# 6장의 주요 내용을 요약한 파일: 분류(Classification)를 위한 GPT 미세 조정
import requests
import zipfile
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import pandas as pd
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# 이전 챕터나 별도 모듈에서 정의된 GPT 모델 관련 함수들 임포트
from previous_chapters import GPTModel, load_gpt2_model

# -----------------------------------------------------------------------------
# 1. 데이터 준비 유틸리티 함수들
# -----------------------------------------------------------------------------

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    """
    스팸 데이터셋(SMS Spam Collection)을 다운로드하고 압축을 해제하는 함수
    """
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    # 파일 다운로드 (스트리밍 방식)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    # 압축 해제
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # 압축 해제된 파일에 .tsv 확장자 추가 (Pandas로 읽기 편하게)
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")


def create_balanced_dataset(df):
    """
    데이터 불균형 해결을 위한 함수.
    스팸(spam) 데이터 수에 맞춰 햄(ham, 정상 메일) 데이터를 언더샘플링합니다.
    """
    # "spam" 라벨의 개수 계산
    num_spam = df[df["Label"] == "spam"].shape[0]

    # "ham" 데이터 중에서 "spam" 개수만큼만 무작위 추출
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # 두 데이터셋 병합
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df


def random_split(df, train_frac, validation_frac):
    """
    데이터셋을 학습(Train), 검증(Validation), 테스트(Test) 셋으로 분할하는 함수
    """
    # 전체 데이터 섞기
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # 분할 지점(인덱스) 계산
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # 데이터 분할
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


class SpamDataset(Dataset):
    """
    PyTorch Dataset 클래스 정의.
    텍스트를 토큰화하고, 패딩(Padding) 처리를 수행합니다.
    """
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # 1. 텍스트 데이터를 토큰 ID 리스트로 변환 (Tokenization)
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        # 2. 최대 길이(max_length) 설정
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # max_length보다 긴 문장은 잘라냄 (Truncation)
            # TODO: 시퀀스를 최대 길이로 잘라낼 끝 인덱스를 채우세요.
            # 힌트: 설정된 최대 길이 `self.max_length`로 슬라이싱하면 됩니다.
            self.encoded_texts = [
                encoded_text[:????]
                for encoded_text in self.encoded_texts
            ]

        # 3. 패딩(Padding): 모든 시퀀스 길이를 max_length로 맞춤
        # GPT-2의 <|endoftext|> 토큰 ID인 50256을 패딩 값으로 사용
        # TODO: 패딩에 사용할 토큰 ID를 채우세요.
        # 힌트: 함수 파라미터로 전달된 `pad_token_id`를 사용합니다.
        self.encoded_texts = [
            encoded_text + [????] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        # TODO: 레이블을 가져올 DataFrame 컬럼명을 채우세요.
        # 힌트: 데이터셋의 정답 컬럼 이름은 'Label'입니다.
        label = self.data.iloc[index]["????"]
        # 텐서 형태로 반환 (입력 데이터, 정답 라벨)
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


# -----------------------------------------------------------------------------
# 2. 평가 및 학습 관련 함수들
# -----------------------------------------------------------------------------

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """
    데이터 로더를 순회하며 모델의 정확도(Accuracy)를 계산
    """
    model.eval()  # 평가 모드 설정 (Dropout 비활성화 등)
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                # GPT 모델은 시퀀스를 출력하므로, 분류를 위해서는 '마지막 토큰'의 출력만 사용
                # logits shape: [batch_size, seq_len, vocab_size 또는 num_classes]
                logits = model(input_batch)[:, -1, :]

            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    단일 배치에 대한 손실(Loss) 계산
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # TODO: 분류에 사용할 시퀀스 내 위치 인덱스를 채우세요.
    # 힌트: GPT는 시퀀스의 마지막 토큰 출력만 분류 예측에 사용합니다.
    logits = model(input_batch)[:, ????, :]
    # Cross Entropy Loss 계산
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    데이터 로더 전체에 대한 평균 손실 계산
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    현재 모델의 Train Loss와 Validation Loss를 평가
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    """
    메인 학습 루프 함수
    """
    # 손실과 정확도 기록용 리스트
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()  # 학습 모드 설정

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 이전 기울기 초기화
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # 역전파 (Gradient 계산)
            optimizer.step()  # 가중치 업데이트

            examples_seen += input_batch.shape[0]
            global_step += 1

            # 주기적으로 평가 수행 (eval_freq 스텝마다)
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 에폭이 끝날 때마다 정확도 계산 및 출력
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    """
    학습 결과(Loss, Accuracy)를 그래프로 시각화하여 저장
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # 에폭(Epoch) 기준 그래프
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # 처리한 예제 수(Examples seen) 기준 보조 축 추가
    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.savefig(f"outputs/{label}-plot.pdf")
    # plt.show()
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — 핵심 코드 골격

원본 Cell `001`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
# -----------------------------------------------------------------------------
# 3. 메인 실행 블록
# -----------------------------------------------------------------------------

test_mode = False
########################################
# 1단계: 데이터셋 다운로드 및 준비
########################################

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "datas/sms_spam_collection.zip"
extracted_path = "datas/sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

try:
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
except (requests.exceptions.RequestException, TimeoutError) as e:
    print(f"Primary URL failed: {e}. Trying backup URL...")
    url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

# 데이터 로드 및 전처리
df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
balanced_df = create_balanced_dataset(df) # 50:50 균형 데이터셋 생성
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1}) # 라벨 인코딩

# Train/Val/Test 분할 및 CSV 저장
train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
train_df.to_csv("datas/train.csv", index=None)
validation_df.to_csv("datas/validation.csv", index=None)
test_df.to_csv("datas/test.csv", index=None)

########################################
# 2단계: 데이터 로더 생성
########################################
tokenizer = tiktoken.get_encoding("gpt2")

# 학습 데이터셋 생성
train_dataset = SpamDataset(
    csv_file="datas/train.csv",
    max_length=None, # 가장 긴 문장에 맞춰 자동 설정
    tokenizer=tokenizer
)

# 검증/테스트 데이터셋은 학습 데이터셋의 max_length를 따름 (일관성 유지)
val_dataset = SpamDataset(
    csv_file="datas/validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

test_dataset = SpamDataset(
    csv_file="datas/test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

num_workers = 0
batch_size = 8
torch.manual_seed(123)

# DataLoader 생성
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True, # 학습 데이터는 섞음
    num_workers=num_workers,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

########################################
# 3단계: 사전 학습된 모델 로드 (Pre-trained Model Loading)
########################################

if test_mode:
    # 테스트 모드일 때는 아주 작은 더미 모델 생성
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 120,
        "drop_rate": 0.0,
        "qkv_bias": False,
        "emb_dim": 12,
        "n_layers": 1,
        "n_heads": 2
    }
    model = GPTModel(BASE_CONFIG)
    model.eval()
    device = "cpu"

else:
    # 실제 GPT-2 Small (124M 파라미터) 모델 설정 및 가중치 로드
    CHOOSE_MODEL = "gpt2-small (124M)"

    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        # 다른 모델 사이즈 설정들...
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    # 데이터셋의 길이가 모델의 컨텍스트 길이를 초과하는지 확인
    assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
        f"Dataset length {train_dataset.max_length} exceeds model's context "
        f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
        f"`max_length={BASE_CONFIG['context_length']}`"
    )

    model_name = "gpt2-small-124M.pth"
    model = load_gpt2_model(model_name, BASE_CONFIG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################
# 4단계: 모델 수정 및 미세 조정 설정 (Fine-tuning Setup)
########################################

# 1. 모든 파라미터를 고정(Freeze)하여 학습되지 않게 설정
# TODO: 파라미터 동결을 위한 값을 채우세요.
# 힌트: 파라미터를 학습에서 제외하려면 requires_grad를 False로 설정합니다.
for param in model.parameters():
    param.requires_grad = ????

torch.manual_seed(123)

# 2. 출력층(Head) 교체: 기존 50257개 단어 예측 -> 2개 클래스(스팸/햄) 예측
# 이 새로운 층은 기본적으로 requires_grad=True 상태임
num_classes = 2
# region [출력층(Head) 교체]
# TODO: 출력층의 클래스 수를 채우세요.
# 힌트: 이진 분류(스팸/햄) 문제이므로 위에서 정의한 num_classes를 사용합니다.
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=????)
# endregion

model.to(device)

# 3. 추가적으로 마지막 트랜스포머 블록과 정규화 층을 학습 가능하도록 잠금 해제 (Unfreeze)
# -> 이렇게 하면 출력층과 모델의 마지막 부분만 데이터에 맞게 조정됨 (효율적 학습)
# region [마지막 트랜스포머 블록과 정규화 층 잠금 해제]
# TODO: 학습할 마지막 블록 인덱스와 학습 허용 값을 채우세요.
# 힌트: 마지막 트랜스포머 블록만 학습하려면 인덱스 -1과 requires_grad = True를 사용합니다.
for param in model.trf_blocks[????].parameters():
    param.requires_grad = ????

# TODO: 정규화 층 학습 허용 값을 채우세요.
# 힌트: requires_grad를 True로 설정하면 해당 파라미터가 학습 대상이 됩니다.
for param in model.final_norm.parameters():
    param.requires_grad = ????
# endregion

########################################
# 5단계: 모델 미세 조정 (Fine-tuning) 실행
########################################

start_time = time.time()
torch.manual_seed(123)

# 옵티마이저 설정 (학습 가능한 파라미터만 전달됨)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

num_epochs = 5
# 학습 함수 호출
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5,
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

########################################
# 6단계: 결과 시각화
########################################

# Loss 그래프 그리기
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

# Accuracy 그래프 그리기
epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

## LLM Practice 06+. LoRA Classification 코드 학습

원본: `llm_hands_on/Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb`
실습본: `practice_notebooks/language/06-lora-classification.ipynb`

### Drill 1 — 핵심 코드 골격

원본 Cell `000`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
import math
import time
import requests
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt # 그래프 저장을 위해 추가

# 로컬 모듈 임포트 (previous_chapters.py가 필요합니다)
from previous_chapters import (
    download_and_unzip_spam_data,
    create_balanced_dataset,
    random_split,
    SpamDataset,
    GPTModel,
    download_model,
    calc_accuracy_loader,
    train_classifier_simple,
    plot_values # 그래프 그리기 함수 추가
)
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — 핵심 코드 골격

원본 Cell `001`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
# -----------------------------------------------------------------------------
# 1. 설정 및 하이퍼파라미터
# -----------------------------------------------------------------------------
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
    "emb_dim": 768, "n_layers": 12, "n_heads": 12
}

BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 8e-4
LORA_RANK = 16
LORA_ALPHA = 16
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 3 — 핵심 코드 골격

원본 Cell `002`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
###중요: LoRA 구현 부분
# -----------------------------------------------------------------------------
# 2. LoRA 클래스 및 함수 정의
# -----------------------------------------------------------------------------
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        """
        LoRA(Low-Rank Adaptation) 레이어 초기화
        """
        super().__init__()
        # TODO: LoRA A 행렬의 rank 차원을 채우세요.
        # 힌트: A 행렬 크기는 (in_dim, rank)입니다.
        self.A = nn.Parameter(torch.empty(in_dim, ????))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        # TODO: LoRA B 행렬의 rank 차원을 채우세요.
        # 힌트: B 행렬 크기는 (rank, out_dim)입니다.
        self.B = nn.Parameter(torch.zeros(????, out_dim))

        self.alpha = alpha
        self.rank = rank

    def forward(self, x):
        # region [LoRA 어댑터 적용]
        # TODO: LoRA forward 식의 파라미터명을 채우세요.
        # 힌트: alpha/rank 스케일과 A, B 행렬곱 순서를 맞추세요.
        x = (self.???? / self.rank) * (x @ self.???? @ self.????)
        # endregion
        return x

class LinearWithLoRA(nn.Module):
    """
    기존의 Linear 레이어를 감싸서 LoRA 어댑터를 추가한 클래스
    """
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        # region [기존 Linear와 LoRA 출력 합산]
        # TODO: LoRA 잔차 호출 대상을 채우세요.
        # 힌트: 기존 linear 출력에 lora(x)를 더해 반환합니다.
        return self.linear(x) + self.????(x)
        # endregion

def replace_linear_with_lora(model, rank, alpha):
    """
    모델 내의 모든 Linear 레이어를 찾아 LoRA가 적용된 레이어로 교체
    """
    for name, module in model.named_children():
        # TODO: LoRA 교체 대상 레이어 타입을 채우세요.
        # 힌트: 대상은 torch.nn.Linear 레이어입니다.
        if isinstance(module, torch.nn.????):
            # 기존 Linear 레이어를 LinearWithLoRA로 교체
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # 재귀 호출
            replace_linear_with_lora(module, rank, alpha)
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 4 — 핵심 코드 골격

원본 Cell `003`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
def main():
    # 시드 설정
    torch.manual_seed(123)

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # --- 데이터셋 준비 ---
    print("데이터셋 준비 중...")
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "datas/sms_spam_collection.zip"
    extracted_path = "datas/sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    try:
        download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    except (requests.exceptions.RequestException, TimeoutError):
        # 백업 URL 시도
        url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
        download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

    # 임시 파일 저장 (Dataset 클래스 사용을 위해)
    train_df.to_csv("datas/train.csv", index=None)
    validation_df.to_csv("datas/validation.csv", index=None)
    test_df.to_csv("datas/test.csv", index=None)

    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = SpamDataset("datas/train.csv", max_length=None, tokenizer=tokenizer)
    val_dataset = SpamDataset("datas/validation.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)
    test_dataset = SpamDataset("datas/test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, drop_last=False)

    # --- 모델 준비 ---
    print("모델 로드 중...")

    file_name = "gpt2-small-124M.pth"
    model_path = f"./models/gpt2/{file_name}"
    download_model(file_name, model_path)

    # Policy Model (학습 대상)
    model = GPTModel(BASE_CONFIG)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.to(device)

    # 출력 헤드 교체 (분류용)
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=768, out_features=num_classes)

    # --- LoRA 적용 ---
    print("LoRA 레이어 적용 및 파라미터 동결 중...")
    # 1. 모든 파라미터 동결
    # TODO: 파라미터 학습 여부 값을 채우세요.
    # 힌트: 전체 동결 단계에서는 False를 설정합니다.
    for param in model.parameters():
        param.requires_grad = ????

    # 2. Linear 레이어를 LoRA로 교체 (LoRA 파라미터는 새로 생성되므로 requires_grad=True 상태)
    # TODO: LoRA 주입 헬퍼 함수명을 채우세요.
    # 힌트: 모델의 Linear 레이어를 LoRA 버전으로 교체하는 함수를 호출합니다.
    ????(model, rank=LORA_RANK, alpha=LORA_ALPHA)

    model.to(device)

    # 학습 가능 파라미터 확인
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"총 학습 가능 LoRA 파라미터 수: {total_params:,}")

    # --- 학습 실행 ---
    print("학습 시작...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)

    start_time = time.time()
    # 반환값 캡처 (loss 및 accuracy 기록)
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=NUM_EPOCHS, eval_freq=50, eval_iter=5,
    )
    end_time = time.time()
    print(f"학습 완료 (소요 시간: {(end_time - start_time) / 60:.2f}분)")

    # --- 결과 시각화 (Loss) ---
    print("학습 결과 시각화 중...")
    epochs_tensor = torch.linspace(0, NUM_EPOCHS, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

    # plot_values 함수 호출
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses, label="loss")

    # --- 최종 평가 ---
    print("최종 모델 평가 중...")
    train_acc = calc_accuracy_loader(train_loader, model, device)
    val_acc = calc_accuracy_loader(val_loader, model, device)
    test_acc = calc_accuracy_loader(test_loader, model, device)

    print(f"훈련 정확도: {train_acc*100:.2f}%")
    print(f"검증 정확도: {val_acc*100:.2f}%")
    print(f"테스트 정확도: {test_acc*100:.2f}%")
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

## LLM Practice 07. Instruction Fine-tuning 코드 학습

원본: `llm_hands_on/Chapter_7_Exercise_Follow_Instructions.ipynb`
실습본: `practice_notebooks/language/07-instruction-finetuning.ipynb`

### Drill 1 — 핵심 코드 골격

원본 Cell `000`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# 7장의 코드를 기반으로 한 최소한의 Instruction Finetuning 예제 파일

from functools import partial
from importlib.metadata import version
import json
import os
import re
import time

import matplotlib.pyplot as plt
import requests
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 로컬 파일(이전 챕터에서 작성된 유틸리티 함수들)에서 필요한 모듈 임포트
# GPTModel 구조, 생성 함수, 학습 루프 함수 등을 가져옵니다.
from previous_chapters import (
    calc_loss_loader,
    generate,
    GPTModel,
    text_to_token_ids,
    train_model_simple,
    token_ids_to_text,
    load_gpt2_model
)

class InstructionDataset(Dataset):
    """
    지시사항(Instruction) 데이터셋을 처리하는 PyTorch Dataset 클래스.
    입력 데이터(지시+입력)와 정답 데이터(응답)를 하나의 텍스트로 합쳐서 토큰화합니다.
    """
    def __init__(self, data, tokenizer):
        self.data = data

        # 텍스트 데이터 미리 토큰화 (Pre-tokenize)
        self.encoded_texts = []
        for entry in data:
            # 포맷팅 함수를 이용해 "지시사항 + 입력" 텍스트 생성
            instruction_plus_input = format_input(entry)
            # 정답(Response) 텍스트 생성
            response_text = f"\n\n### Response:\n{entry['output']}"

            # 모델은 이 전체 텍스트(질문+답변)를 보고 다음 토큰을 예측하도록 학습됨
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """
    데이터 로더에서 배치를 만들 때 사용하는 커스텀 함수.
    가변 길이의 시퀀스를 배치의 최대 길이에 맞춰 패딩(padding)하고,
    정답(target) 데이터에서 패딩 부분은 손실(loss) 계산에서 제외하도록 처리합니다.
    """
    # 배치 내에서 가장 긴 시퀀스 길이 계산 (패딩을 위해 +1 여유)
    batch_max_length = max(len(item)+1 for item in batch)

    # 입력(inputs)과 정답(targets) 리스트 준비
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # 문장 끝 토큰 <|endoftext|> 추가
        # TODO: 시퀀스 끝에 추가할 토큰 값을 채우세요.
        # 힌트: 패딩 전에 구분 토큰이 필요한 경우 pad_token_id를 추가합니다.
        new_item += [????]

        # 가장 긴 길이에 맞춰 패딩 추가
        # TODO: 패딩 토큰 값을 채우세요.
        # 힌트: 모든 샘플 길이를 맞추기 위해 pad_token_id로 채웁니다.
        padded = new_item + [????] * (batch_max_length - len(new_item))

        # TODO: 입력과 타깃 슬라이싱 인덱스를 채우세요.
        # 힌트: 다음 토큰 예측에서는 inputs=padded[:-1], targets=padded[1:] 형태를 만듭니다.
        # 입력은 마지막 토큰 제외 (0 ~ n-1)
        inputs = torch.tensor(padded[:????])
        # 정답은 첫 번째 토큰 제외 (1 ~ n) -> 다음 토큰 예측 문제이므로
        targets = torch.tensor(padded[????:])

        # 중요: 패딩 부분 마스킹 처리
        # 타겟에서 패딩 토큰인 부분은 ignore_index(-100)로 바꿔서 CrossEntropyLoss 계산 시 무시되게 함
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()

        # 패딩이 시작되는 지점 이후의 모든 타겟 값을 -100으로 설정
        # region [패딩 마스킹 처리]
        # TODO: 패딩 무시 시작 인덱스를 채우세요.
        # 힌트: 첫 번째 패딩은 유지하고 그 뒤의 패딩부터 ignore_index로 설정합니다.
        if indices.numel() > 1:
            targets[indices[????:]] = ignore_index
        # endregion

        # 선택적으로 최대 시퀀스 길이 제한 (메모리 관리 등 목적)
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 텐서로 변환 및 디바이스(GPU/CPU)로 이동
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def download_and_load_file(file_path, url):
    """
    데이터 파일이 없으면 다운로드하고, JSON 형태로 로드하는 유틸리티 함수
    """
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


def format_input(entry):
    """
    Alpaca 스타일의 프롬프트 템플릿을 적용하는 함수.
    모델에게 역할을 부여하고 입력 형식을 통일합니다.
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    # 추가적인 입력 정보(Context 등)가 있는 경우 추가
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    학습 진행 상황(Loss)을 그래프로 시각화하여 PDF로 저장하는 함수
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 에폭(Epoch) 기준 Loss 그래프
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # 처리한 토큰 수(Tokens seen) 기준 보조 x축 생성
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plot_name = "outputs/loss-plot-standalone.pdf"
    print(f"Plot saved as {plot_name}")
    plt.savefig(plot_name)
    # plt.show()


def main(test_mode=False):
    #######################################
    # 패키지 버전 출력 (디버깅용)
    #######################################
    print()
    pkgs = [
        "matplotlib",  # 시각화
        "tiktoken",    # 토크나이저
        "torch",       # 딥러닝 프레임워크
        "tqdm",        # 진행률 표시 바
    ]
    for p in pkgs:
        print(f"{p} version: {version(p)}")
    print(50*"-")

    #######################################
    # 1. 데이터셋 다운로드 및 준비
    #######################################
    file_path = "datas/instruction-data.json"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    data = download_and_load_file(file_path, url)

    # 데이터 분할 (Train 85% / Test 10% / Valid 5%)
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    # 테스트 모드일 경우 아주 적은 데이터만 사용 (빠른 실행 확인용)
    if test_mode:
        train_data = train_data[:10]
        val_data = val_data[:10]
        test_data = test_data[:10]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    print(50*"-")

    # 토크나이저 및 디바이스 설정
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(50*"-")

    # 커스텀 collate 함수 설정 (partial을 사용해 고정 인자 전달)
    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    # 학습용 데이터 로더
    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn, # 커스텀 배치 처리 함수 사용
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    # 검증용 데이터 로더
    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    #######################################
    # 2. 사전 학습된 모델 로드 (Pre-trained Model)
    #######################################

    if test_mode:
        # 테스트 모드용 초소형 더미 모델
        BASE_CONFIG = {
            "vocab_size": 50257,
            "context_length": 120,
            "drop_rate": 0.0,
            "qkv_bias": False,
            "emb_dim": 12,
            "n_layers": 1,
            "n_heads": 2
        }
        model = GPTModel(BASE_CONFIG)
        model.eval()
        device = "cpu"
        CHOOSE_MODEL = "Small test model"

    else:
        # 실제 학습용 설정: GPT-2 Medium (355M)
        BASE_CONFIG = {
            "vocab_size": 50257,
            "context_length": 1024,
            "drop_rate": 0.0,
            "qkv_bias": True
        }

        model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        CHOOSE_MODEL = "gpt2-medium (355M)"
        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

        # 로컬에 저장된 모델 가중치 파일 로드
        model_name = "gpt2-medium-355M.pth"
        model = load_gpt2_model(model_name, BASE_CONFIG)

        model.eval()
        model.to(device)

    print("Loaded model:", CHOOSE_MODEL)
    print(50*"-")

    #######################################
    # 3. 모델 미세 조정 (Finetuning)
    #######################################
    print("Initial losses")
    # 학습 전 초기 손실값 확인
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("   Training loss:", train_loss)
    print("   Validation loss:", val_loss)

    start_time = time.time()
    # 옵티마이저 설정 (AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    num_epochs = 2

    torch.manual_seed(123)

    # 실제 학습 수행 (previous_chapters.py에 정의된 함수 사용)
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # 학습 결과 시각화
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    print(50*"-")

    #######################################
    # 4. 결과 저장 및 테스트
    #######################################
    print("Generating responses")
    # 테스트 데이터셋에 대해 모델 응답 생성
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

        input_text = format_input(entry)

        # 모델 생성 (Inference)
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)

        # 프롬프트 부분을 제거하고 응답 부분만 추출
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        test_data[i]["model_response"] = response_text

    # 생성된 응답을 포함하여 JSON 파일로 저S장
    test_data_path = "outputs/instruction-data-with-response-standalone.json"
    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)
    print(f"Responses saved as {test_data_path}")

    # 미세 조정된 모델 가중치 저장
    file_name = f"outputs/{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft-standalone.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

## LLM Practice 07+. DPO 코드 학습

원본: `llm_hands_on/Chapter_7_Exercise_Follow_Instructions_dpo.ipynb`
실습본: `practice_notebooks/language/07-dpo.ipynb`

### Drill 1 — 핵심 코드 골격

원본 Cell `000`. 이 셀은 **핵심 코드 골격** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
import os
import json
import time
import requests
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from functools import partial
import tiktoken

# ==========================================
# [중요] 외부 모듈 의존성 처리
# 이 코드는 책이나 튜토리얼의 일부로, 이전 챕터에서 정의된 모델(GPTModel)이 필요합니다.
# ==========================================
try:
    from previous_chapters import GPTModel, generate_and_print_sample, download_model
except ImportError:
    print("경고: 'previous_chapters' 모듈을 찾을 수 없습니다. GPTModel 등의 정의가 필요합니다.")

# ==========================================
# 1. 설정 및 데이터 로드 유틸리티
# 데이터를 인터넷에서 다운로드하거나 로컬에서 불러오는 기능을 담당합니다.
# ==========================================
def download_and_load_file(file_path, url):
    """
    지정된 경로에 파일이 없으면 URL에서 다운로드하고,
    파일을 읽어 JSON 데이터를 반환합니다.
    """
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(response.text)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    return json.loads(text_data)


def format_input(entry):
    """
    JSON 데이터의 항목을 모델이 이해할 수 있는 프롬프트 형식으로 변환합니다.
    (알파카(Alpaca) 데이터셋 스타일 포맷팅)
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    # 추가 입력(Input)이 있는 경우에만 붙임
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

# ==========================================
# 2. 데이터셋 및 Collate 함수 (DPO용)
# DPO는 (프롬프트, 선택된 답변, 거부된 답변)의 쌍이 필요합니다.
# ==========================================
class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []

        # 데이터를 미리 토큰화(Tokenization)하여 저장합니다.
        for entry in data:
            prompt = format_input(entry)
            rejected_response = entry["rejected"] # 👎 덜 선호되는 답변
            chosen_response = entry["chosen"]     # 👍 더 선호되는 답변

            # 프롬프트 부분만 따로 인코딩 (나중에 마스킹하기 위해 필요)
            prompt_tokens = tokenizer.encode(prompt)

            # 프롬프트 + 답변 형태로 전체 문장을 인코딩
            # TODO: 선호/비선호 응답 변수명을 채우세요.
            # 힌트: 바로 위에서 만든 chosen_response와 rejected_response를 Response 뒤에 붙입니다.
            # 예시: chosen 쪽은 chosen_response, rejected 쪽은 rejected_response를 사용합니다.
            chosen_full_tokens = tokenizer.encode(f"{prompt}\n\n### Response:\n{????}")
            rejected_full_tokens = tokenizer.encode(f"{prompt}\n\n### Response:\n{????}")

            self.encoded_texts.append({
                "prompt": prompt_tokens,
                "chosen": chosen_full_tokens,
                "rejected": rejected_full_tokens,
            })

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

def custom_collate_fn(batch, pad_token_id=50256, allowed_max_length=None, mask_prompt_tokens=True, device="cpu"):
    """
    배치(Batch) 내의 데이터 길이를 맞추고(Padding), 마스크(Mask)를 생성하는 함수입니다.
    """
    batch_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "rejected_mask": [], # Loss 계산 시 무시할 부분 (패딩 등)
        "chosen_mask": []
    }

    # 1. 배치 내에서 가장 긴 시퀀스 길이 찾기 (패딩을 위해)
    max_length_common = 0
    if batch:
        for key in ["chosen", "rejected"]:
            current_max = max(len(item[key]) + 1 for item in batch)
            max_length_common = max(max_length_common, current_max)

    # 2. 패딩 추가 및 마스크 생성
    for item in batch:
        prompt = torch.tensor(item["prompt"])
        batch_data["prompt"].append(prompt)

        for key in ["chosen", "rejected"]:
            sequence = item[key]
            # 최대 길이에 맞춰 패딩 토큰 추가
            padded = sequence + [pad_token_id] * (max_length_common - len(sequence))

            # 기본 마스크: 데이터가 있는 곳은 1(True), 패딩은 0(False)
            mask = torch.ones(len(padded)).bool()
            mask[len(sequence):] = False  # 패딩 부분 마스킹

            # [중요] 프롬프트 부분 마스킹 (선택적)
            # DPO는 '답변'의 확률 차이를 학습하므로, 질문(Prompt) 부분은 Loss 계산에서 제외합니다.
            # +2는 \n\n때문
            if mask_prompt_tokens:
                mask[:prompt.shape[0] + 2] = False

            batch_data[key].append(torch.tensor(padded))
            batch_data[f"{key}_mask"].append(mask)

    # 3. 텐서 변환 및 디바이스(GPU/CPU) 이동
    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        tensor_stack = torch.stack(batch_data[key])
        # 길이 제한이 있다면 자르기
        if allowed_max_length is not None:
            tensor_stack = tensor_stack[:, :allowed_max_length]
        batch_data[key] = tensor_stack.to(device)

    return batch_data

# ==========================================
# 3. DPO Loss 및 Log Probability 계산 함수
# DPO의 핵심 수학적 로직이 들어있는 부분입니다.
# ==========================================
def compute_logprobs(logits, labels, selection_mask=None):
    """
    모델의 출력(logits)과 정답(labels)을 받아 해당 정답 토큰의 로그 확률을 계산합니다.
    """
    # Auto-regressive 모델 특성상, 입력 [A, B, C]에 대해 예측은 [B, C, D]가 되므로 시프트(Shift)합니다.
    # TODO: 자동회귀 시프트 인덱스를 채우세요.
    # 힌트: labels는 1칸 앞당기고 logits는 마지막 시점을 제외해 정렬합니다.
    labels = labels[:, ????:].clone()
    logits = logits[:, :????, :]

    log_probs = F.log_softmax(logits, dim=-1)

    # 실제 정답 레이블에 해당하는 확률값만 추출 (gather 사용)
    # TODO: gather의 input 텐서를 채우세요.
    # 힌트: raw logits가 아닌 log_probs에서 정답 위치를 gather해야 합니다.
    selected_log_probs = torch.gather(
        input=????,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    if selection_mask is not None:
        # 마스크도 시프트하여 적용 (패딩이나 프롬프트 영역 무시)
        mask = selection_mask[:, 1:].clone()
        selected_log_probs = selected_log_probs * mask

        # 유효한 토큰들의 로그 확률 평균 계산
        avg_log_prob = selected_log_probs.sum(-1) / mask.sum(-1)
        return avg_log_prob
    else:
        return selected_log_probs.mean(-1)

def compute_dpo_loss(model_chosen_logprobs, model_rejected_logprobs,
                     reference_chosen_logprobs, reference_rejected_logprobs, beta=0.1):
    """
    DPO 손실 함수 계산:
    Policy 모델이 Reference 모델보다 'chosen' 답변을 더 선호하고, 'rejected' 답변을 덜 선호하도록 유도합니다.

    beta: Reference 모델에서 얼마나 벗어날지 제어하는 하이퍼파라미터 (보통 0.1~0.5)
    """
    # 모델의 (Chosen - Rejected) 로그 확률 차이
    # TODO: 정책 모델 chosen 로그확률 변수를 채우세요.
    # 힌트: chosen - rejected 순서로 빼야 preference 방향이 맞습니다.
    model_logratios = ???? - model_rejected_logprobs
    # 기준 모델의 (Chosen - Rejected) 로그 확률 차이
    # TODO: 레퍼런스 모델 chosen 로그확률 변수를 채우세요.
    # 힌트: reference도 chosen - rejected 순서를 동일하게 유지합니다.
    reference_logratios = ???? - reference_rejected_logprobs

    # 두 비율의 차이 (Policy가 Reference보다 얼마나 더 잘 구분했는가)
    # TODO: DPO logits 차감 대상을 채우세요.
    # 힌트: policy 비율에서 reference 비율을 빼는 형태입니다.
    logits = model_logratios - ????

    # Sigmoid 후 음수 로그 (Cross Entropy와 유사) -> 이 값을 최소화하면 선호도 차이가 극대화됨
    # TODO: DPO 손실 함수명을 채우세요.
    # 힌트: beta * logits에 logsigmoid를 적용해 손실을 계산합니다.
    losses = -F.????(beta * logits)

    # 학습 추적용 보상(Reward) 계산 (실제 학습엔 안 쓰이고 로깅용)
    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()

    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

def compute_dpo_loss_batch(batch, policy_model, reference_model, beta):
    """
    하나의 배치에 대해 전체 DPO 과정을 수행하는 헬퍼 함수
    """
    # 1. 학습 중인 모델(Policy Model)의 로그 확률 계산 (기울기 계산 O)
    policy_chosen_log_probas = compute_logprobs(
        logits=policy_model(batch["chosen"]),
        labels=batch["chosen"],
        selection_mask=batch["chosen_mask"]
    )
    policy_rejected_log_probas = compute_logprobs(
        logits=policy_model(batch["rejected"]),
        labels=batch["rejected"],
        selection_mask=batch["rejected_mask"]
    )

    # 2. 기준 모델(Reference Model)의 로그 확률 계산 (기울기 계산 X -> 메모리 절약)
    with torch.no_grad():
        ref_chosen_log_probas = compute_logprobs(
            logits=reference_model(batch["chosen"]),
            labels=batch["chosen"],
            selection_mask=batch["chosen_mask"]
        )
        ref_rejected_log_probas = compute_logprobs(
            logits=reference_model(batch["rejected"]),
            labels=batch["rejected"],
            selection_mask=batch["rejected_mask"]
        )

    # 3. 최종 Loss 계산
    loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
        model_chosen_logprobs=policy_chosen_log_probas,
        model_rejected_logprobs=policy_rejected_log_probas,
        reference_chosen_logprobs=ref_chosen_log_probas,
        reference_rejected_logprobs=ref_rejected_log_probas,
        beta=beta
    )
    return loss, chosen_rewards, rejected_rewards

# ==========================================
# 4. 평가 및 학습 루프
# 실제 모델 학습을 돌리는 메인 루프입니다.
# ==========================================
def evaluate_dpo_loss_loader(policy_model, reference_model, train_loader, val_loader, beta, eval_iter):
    """
    학습 중간에 모델 성능을 평가(Validation)하는 함수
    """
    policy_model.eval() # 평가 모드 전환 (Dropout 등 비활성화)
    with torch.no_grad():
        # 로더를 순회하며 평균 Loss 계산하는 내부 함수
        def compute_loader_metric(loader):
            total_loss, total_chosen, total_rejected = 0., 0., 0.
            num_batches = min(eval_iter, len(loader))
            if num_batches == 0: return float("nan"), float("nan"), float("nan")

            for i, batch in enumerate(loader):
                if i >= num_batches: break
                loss, chosen, rejected = compute_dpo_loss_batch(batch, policy_model, reference_model, beta)
                total_loss += loss.item()
                total_chosen += chosen.item()
                total_rejected += rejected.item()
            return total_loss/num_batches, total_chosen/num_batches, total_rejected/num_batches

        train_loss, train_chosen, train_rejected = compute_loader_metric(train_loader)
        val_loss, val_chosen, val_rejected = compute_loader_metric(val_loader)

    policy_model.train() # 다시 학습 모드로 전환
    return {
        "train_loss": train_loss,
        "train_chosen_reward": train_chosen,
        "train_rejected_reward": train_rejected,
        "val_loss": val_loss,
        "val_chosen_reward": val_chosen,
        "val_rejected_reward": val_rejected
    }

def train_model_dpo_simple(policy_model, reference_model, train_loader, val_loader,
                           optimizer, num_epochs, beta, eval_freq, eval_iter, start_context, tokenizer):
    """
    전체 학습 과정을 관리하는 함수
    """
    tracking = {"train_losses": [], "val_losses": [], "tokens_seen": []}
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        policy_model.train() # 학습 모드 시작
        for batch in train_loader:
            optimizer.zero_grad() # 이전 기울기 초기화

            # Loss 계산
            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch, policy_model=policy_model, reference_model=reference_model, beta=beta
            )

            loss.backward() # 역전파 (기울기 계산)
            optimizer.step() # 가중치 업데이트

            tokens_seen += batch["chosen"].numel()
            global_step += 1

            # 평가 주기마다 성능 측정 및 출력
            if global_step % eval_freq == 0:
                res = evaluate_dpo_loss_loader(
                    policy_model, reference_model, train_loader, val_loader, beta, eval_iter
                )
                tracking["train_losses"].append(res["train_loss"])
                tracking["val_losses"].append(res["val_loss"])
                tracking["tokens_seen"].append(tokens_seen)

                # Margin = Chosen Reward - Rejected Reward (클수록 좋음, 모델이 정답을 오답보다 더 선호한다는 뜻)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {res['train_loss']:.3f}, Val loss {res['val_loss']:.3f}, "
                      f"Train Margin {res['train_chosen_reward']-res['train_rejected_reward']:.3f}")

        # 에포크가 끝날 때마다 샘플 생성하여 육안으로 확인
        generate_and_print_sample(model=policy_model, tokenizer=tokenizer, device=loss.device, start_context=start_context)

    return tracking
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 2 — 💡 주요 핵심 포인트

원본 Cell `002`. 이 셀은 **💡 주요 핵심 포인트** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
# ==========================================
# 5. 메인 실행 코드
# ==========================================
# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = tiktoken.get_encoding("gpt2")

# 데이터 로드
file_path = "datas/instruction-data-with-preference.json"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/04_preference-tuning-with-dpo/instruction-data-with-preference.json"
data = download_and_load_file(file_path, url)

# 데이터 분할
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

# DataLoader 생성
customized_collate_fn = partial(
    custom_collate_fn, device=device, mask_prompt_tokens=True, allowed_max_length=1024
)

batch_size = 8
train_loader = DataLoader(PreferenceDataset(train_data, tokenizer), batch_size=batch_size,
                            collate_fn=customized_collate_fn, shuffle=True, drop_last=True)
val_loader = DataLoader(PreferenceDataset(val_data, tokenizer), batch_size=batch_size,
                        collate_fn=customized_collate_fn, shuffle=False)

# 모델 설정 (SFT 모델 로드)
BASE_CONFIG = {
    "vocab_size": 50257, "context_length": 1024, "drop_rate": 0.0, "qkv_bias": True,
    "emb_dim": 1024, "n_layers": 24, "n_heads": 16 # gpt2-medium config
}

file_name = "gpt2-medium-355M.pth"
model_path = f"./models/gpt2/{file_name}"
download_model(file_name, model_path)

# Policy Model (학습 대상)
policy_model = GPTModel(BASE_CONFIG)
policy_model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
policy_model.to(device)

# Reference Model (고정 기준점)
reference_model = GPTModel(BASE_CONFIG)
reference_model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
reference_model.to(device)
reference_model.eval()

# 학습 시작
optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-6, weight_decay=0.01)

start_time = time.time()
num_epochs = 1
tracking = train_model_dpo_simple(
    policy_model=policy_model,
    reference_model=reference_model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    num_epochs=num_epochs,
    beta=0.1,
    eval_freq=5,
    eval_iter=5,
    start_context=format_input(val_data[2]),
    tokenizer=tokenizer
)
print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 3 — 💡 주요 핵심 포인트

원본 Cell `003`. 이 셀은 **💡 주요 핵심 포인트** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
from previous_chapters import plot_losses
# 대안:
# from llms_from_scratch.ch05 import plot_losses
num_epochs=1

epochs_tensor = torch.linspace(0, num_epochs, len(tracking["train_losses"]))
plot_losses(
    epochs_seen=epochs_tensor,
    tokens_seen=tracking["tokens_seen"],
    train_losses=tracking["train_losses"],
    val_losses=tracking["val_losses"],
    label="loss"
)
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?

### Drill 4 — 💡 주요 핵심 포인트

원본 Cell `004`. 이 셀은 **💡 주요 핵심 포인트** 단계의 객체 연결과 데이터 흐름을 복원하는 문제다.

```python
from previous_chapters import (
    generate,
    text_to_token_ids,
    token_ids_to_text
)

torch.manual_seed(123)


for entry in test_data[:3]:

    input_text = format_input(entry)

    token_ids = generate(
        model=reference_model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    reference_response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )

    token_ids = generate(
        model=policy_model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    policy_response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )

    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nReference model response:\n>> {reference_response_text.strip()}")
    print(f"\nPolicy model response:\n>> {policy_response_text.strip()}")
    print("\n-------------------------------------\n")
```

**확인 질문**

1. 이 셀의 입력 객체와 출력 객체는 무엇인가?
2. shape 또는 자료구조가 다음 단계에서 어떻게 사용되는가?
3. 핵심 함수·클래스 한 줄을 제거하면 파이프라인 어느 지점이 끊기는가?
