# LLM 

AI 역사 ~ LLM 2일차까지

https://github.com/heury
https://heury.github.io/llm_lecture2/#/2 
https://github.com/heury/llm_hands_on 실습자료 등

## Day1

pg82. 절대위치 삼각함수
-> 위치인코딩의 가까운 관계와 먼 관계표현을 위해 삼각함수 주기를 늘림.(찾아보기)

실습 Ch.1 / Ch.2 진행. Ch.2는 시험범위 O
정답은 `https://github.com/heury/llm_hands_on/tree/exercise_answer` (브랜치 확인)

pg110. Masking
GPT는 미래 토큰 확인 불가. -> Masking(-inf 값을 주어 sigmoid 시 0이 되도록)
양방향 문맥을 볼 수 있는 masked ~~는 미래토큰확인가능

? attention map에서 input과 output이 같은 상황에서의 값은 무슨 의미일지? (나를 입력했을 때 내 뒤에꺼가 나올 확률일 것 같은데. (self-attention))

pg122 RMS Nomalization을 요새 많이 사용함. 이유는 평균을 구하는 게 비용이 크기때문.
(또한, 평균 안 써도 된다는 연구 결과도 있음.)
But, RMS 구하는 것도 Root Mean Square이다보니, 평균에 대해서 어차피 구하는 것으로 보이는데 뭐가 다른지.

## Day2

Ch.5 training 학습과정 중요. plot이나 log남기거나 evaluation은 별로 안 중요
실제 학습하는 부분이 어떻게 되는지, loss는 어떻게 계산하는지, optimizer는 어떻게 선언하는지
모델은 어떻게 저장하고 로드하는지를 잘 이해해야함.

pg177
LoRA 초기화에서 A는 gaussian random, B는 0행렬
첫 순전파에서는 orgW로 한 결과가 나오는데, AxB이고, 
B에대해 편미분하면 A.  A에 대해 편미분하면 0
-> A는 업데이트 X, B는 업데이트.
(내용 확인해봐야함)

Ch.6 finetuning에서 포인트는 param 마지막에 output쪽에서 trf_blocks[-1]의 
param.requires_grad = True로 두는 것.
또한, final_norm.parameters도 requires_grad = True로 두는 것.
(즉, 나머지는 freeze한 상태로 유지하고, 마지막층에 있는 것들만 True)

Ch.6 LoRA도 똑같이 param 다 freeze한 후, 마지막 layer를 LoRA 레이어로 변경시킴.
여기서는 LoRA로 replace를 하더라도, Freeze 된 상태에서 새로운 파라미터가 생성되기에 requires_grad = True상태임.

// 말씀드린대로, 로깅이나 결과 저장같은 것보다는, 말씀드린 코드쪽을 보기
// 학습과정과 관련된 부분. 데이터셋 로드쪽은 앞에서 배웠던 내용이니 크게 중요 X
// 적당히만 알아두기.

Ch.7 지시미세튜닝에서는 padding 시 50256보단 -100으로 함.
안 그러면 endoftext ... 무한히 반복되는 경우 발생.
또한, Response만 추론하면 되기에, 앞에 지시와 입력은 -100으로 Masking
(요새는 이렇게 알파카지침으로는 잘 안하고, 스페셜토큰활용)

Ch.7 실습에서는 -100채우는게 있긴한데, 프롬프트부분은 마스킹이 안 되어있음.
실제 환경에서는 되어있어야함.
Ch7에서는 collate함수와 dataset loader부분만 잘 알면 됨.
DPO쪽에서는 loss함수.

뒤 강의내용은 시험범위 X

---

## 중요한 부분

### Ch.2  
encode하고 decode하는 부분. allowed_special도.  
target은 1 shift되어 [1:context_length]인것도.  

즉, 입력이 있을 때, target은 1만큼 shift되는 것 중요.

DataLoader는 dataset에서 batch로 해서 drop_last(버릴것버리기)
데이터 어떻게 불러들일지 알려주는 애.

embedding layer 만들 때 갯수는 임베딩 객체가 다 담겨있는 것.
vocab_size은 tokenizer에 종속된 것, output_dim은 설계하기 나름.
위치 임베딩은 context_length만큼만 가지면 됨. output_dim이야 token emb와 동일해야.

입력은 [8, 4, 256]크기인데, 위치는 [4, 256]임. broadcasting됨.

토크나이저 비교는 참고만.

### Ch.3

q, k, v 나누기 위해 layer 선언.
d_in은 embedding 차원과 동일. 출력차원은 정의하기 나름.
GPT모델 내부에서는 multi head attention 고려하여 맞춰줘야함.

forward쪽에서 mask있는데 이건 1개만 있으면 되니깐 미리 register_buffer로 잡아둠

q,k,v 통과시키고, qK^T 해서 attention scores 만들어둠.
여기서 마스킹할 때 -torch.inf 넣어둬야 softmax 시 0으로 됨.

-> attention weights = softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
나눠주는 이유는 너무 커지는 것 막기 위해 임베딩차원의 루트로.

이후 context_vec = attn_weights @ values

Multihead attention은 차원 맞추기 위해 head_dim = d_out // num_heads로 해준다는 것만 기억.

### Ch.4 

LayerNorm에서 scale, shift 정의해놔야하고, scale은 1로 해놓고, 시프트는 0(수식상 당연)
norm_x = (x - mean) / torch.sqrt(var + self.eps) eps는 당연히 0으로 안 나누게

FF에서 처음에 입력은 임베딩 차원과 동일해야하고,(잔차연결해서 넘어오니깐 처음인풋과 동일)
출력은 N배 뻥튀기. ex) 768 -> 3072
GELU 돌리고, 다시 돌려놔야하니 3072 -> 768로

TransformerBlock에서는 Attention과 FeedForward가 중요.
교안에서는 Pre-Norm으로.

이 블럭에서 forward가 중요
shortcut <- 잔차연결위해.
x에 정규화하고 어텐션하고 드랍아웃하고 잔차연결.
이후 피드포워드에서 또 
shortcut <- 잔차연결위해 잡아놓고
정규화, 피드포워드, 드랍아웃, 잔차연결.

GPT모델에서는 이제 입려부터 신경쓰니
tok_emb, pos_emb, drop_emb 잡아놓고
블럭 쌓고
마지막 정규화 final_norm = LayerNorm(cfg["emb_dim])과
out_head = nn.Linear(emb_dim , vocab_size)  요기선 vocab_size가 출력인거 중요.

이제 실제 forward할땐 x = tok_embeds + pos_embeds 해놓고 드랍아웃하고 이제 트랜스포머 통과
이후 final_norm해놓고 out_head해서 각 단어에 대한 예측 점수 (logits) 나옴.

모델 추론 전에는 no_grad()해놔서 gradient 계산 안 하도록 해놓기.
이후 예측값인 다음단어를 찾아야하니
logits = logits[:, -1, :]  <-- (batch, n_token, vocab_size) -> (batch, vocab_size)

여기서 가장 확률 높은거 찾아야하니 torch.argmax(logits, dim=-1, keepdim=True)

실제 main에서는 
불러놓고, 토크나이저 정의하고, 근데 여기서 encoded_tensor은 unsqueeze로 배치차원추가.
디코딩할때는 씌워놨으니 squeeze(0)으로 빼기

### Ch.5

여기선 cross entrophy 계산이 중요.
우선, .to(device) 로 메모리에 올려두는것이 다름.
model.train()은 기본적으로 설정은 되어있으나, 중간에 로그를 남기는 과정에서 추론돌려서
eval()로 바뀌기에 다시 train으로 바꿔주기 위해서.

optimizer.zero_grad()는 옵티마이저의 기울기 초기화

추론에서는 torch.no_grad()해놓고 추론돌리고 하는 그런 과정.
그리고 argmax해놓고 torch.cat으로 붙이는 거

input device와 동일하게 model.to(device)동일.

학습 쭉 하고 학습 끝나면 toorch.save로 

### Ch.6

out_features를 num_classes로 하는거랑
마지막 출력층쪽에있는거만 requires_grad = True고
나머지는 다 freeze 해야하니 param.requires_grad = False

LoRA에서는 A행렬 일단 가우시안분포로 초기화해놓고 (in_dim, rank)로 B는 0행렬로.

LoRA 만들어놓고, 이제 replace_linear_with_lora함수에서 교체하는거만들기.
기존 Linear에서 LoRA로 바꾸는 것.

LoRA에서도 나머지는 다 freeze.
for param in model.parameters():
    paam.requires_grad = False

두고나서 replace하면 됨. (어차피 새로생성되는거라 안얼어있음.)

### Ch.7

collate 함수에서 패딩시키는데, 입력은 맨뒤꺼빼고, 타겟은 패딩시작되는거 이후 -100으로.
(즉, Ch.7인 Follow Instructions에서는 -100으로 패딩시키는 것만 잘 생각)

DPO는 데이터에 chosen, rejected만 추가됨. 
Policy모델있고 계산하는 그런 .

계산할 때 아래 내용이 중요.
    model_logratios = model_chosen_logprobs - model_rejected_logprobs
    reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
    logits = model_logratios - reference_logratios

그리고 losses = -F.logsigmoid(beta * logits)로 계산되는 과정이 중요.
