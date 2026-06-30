# LLM 1

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

