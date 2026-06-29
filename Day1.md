# LLM 1

AI 역사 ~ LLM 2일차까지

https://github.com/heury
https://heury.github.io/llm_lecture2/#/2 
https://github.com/heury/llm_hands_on 실습자료 등

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

