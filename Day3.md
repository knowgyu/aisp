# Vision

## Day1

### Intro
- Evolution of Computer Vision
- 지도학습/비지도학습/ 등
- Background Knowledge -> image sensor 원리, 영상신호처리, 수학(행렬, 미분, 확률통계)

### Basic Architecture (DNN_CNN)

**DNN**

Perceptron -> MLP -> DNN -> Back-propagation
(Back-propagation 시 gradient가 대체로 0근처에 몰려있어서 gradient vanishing이 발생함
그래서 normalize가 필요함.)

![alt text](image.png)
logit과 prob은
sigmoid 들어가기 전이 logit, 들어가서 나온 output이 probability.
logit = 승산 = ln(odds), odds = p / (1 - p)

sigmoid function = 위에있는 식 logit식을 l과 p에 관해 정리하면
p = 1 / (1 + e^(-1))

![alt text](image-1.png)
dp/dl = p(1-p) (위 그림상 빨간 점선. 옆으로갈수록 미분값이 0에 가까워지기에 )

softmax 시 max-trick (만약, Output laye의 logit들이 다 크면, overflow남.)
-> m = max(Z_j)라는, logit값 중 가장 큰 값을 m으로 두고, 연산 시 e^(z_i - m)으로 두어 e^m을 미리나눠두기도.

BCE , CE(Cross Entropy)

**CNN**

Convolution (Mirroring -> Moving -> Multiplication -> Addition)
Cross-correlation은 미러링 없이. (사실 CNN은 correlation에 가까움, CNN에서 굳이 미러링을 할 필요가 없음.)
![alt text](image-2.png)

