# For ML/DL/RL/... beginners

**Week01: 딥러닝 들어가기**

1. 수학적 이해: 비선형성, 오차역전파, MLE, 손실 함수, representation learning
2. 왜 단순한 행렬곱과 비선형성만으로는 chatgpt와 같이 복잡한 학습이 어려운가?: inductive bias에 대하여(CNN, RNN, attention mechanism) + Loss surface
3. ANN의 문제를 어떻게 해결했는가?: normalization, regularization, optimizing, architecture

further reading:

-   _[Manifold](./week01/manifold_study.ipynb) Visualization_

![spiral](./week01/docs/spiral.jpg)

다음과 같이 선형 함수로 근사하기 어려운 데이터를 처리하기 위해 필요한 것은 비선형 함수(흔히 activation이라 불림)이다. 이 비선형 함수들의 중첩을 통해서 마지막 헤드가 판별하기 쉽게 데이터의 형태를 변화시키는 과정을 아래 실험에서 확인할 수 있다.

아래는 마지막 선형 변환 이전의 2차원 데이터의 시각화 결과이다. feature을 구별하기 위해서 본래 space의 distortion을 거쳐 다음과 같은 분리가 이루어진 것을 확인할 수 있다.

| 0th                                    | 100th                                    | 1000th                                    |
| -------------------------------------- | ---------------------------------------- | ----------------------------------------- |
| ![image](./week01/docs/manifold_0.jpg) | ![image](./week01/docs/manifold_100.jpg) | ![image](./week01/docs/manifold_1000.jpg) |

우리가 지구에 살때 위도, 경도 두 개의 변수만으로 위치를 표현할 수 있는 것처럼 다음과 같이 복잡한 데이터를 원 좌표계에서 바라보는 것보다 이를 저차원으로 매핑하거나 왜곡된 좌표계에서 바라보는 것이 판단에 용이할 것이다.

-> 데이터의 숨겨진 모양을 찾아나가는 것!

_궁금증을 가질만한 질문들_

1. boundary의 문제: 어딜 경계로 두 데이터가 나누어지는지 (hallucination+bayes optimization)
    - P(Y, X)/P(X) vs P(Y|X)
2. 일반화는 어떻게 이루어지는거지?: test set의 비율을 조정하면서 실험해보기
3. 좌표계를 그렇게 왜곡시켰는데 파라메터 공간에 대해서 일반적인 GD를 쓸 수 있나?: Riemmanian metric+Natural Gradient

-   [Loss surface](./week01/loss_surface.ipynb) visualization of your model! (why residual connection is GOD.)

![loss_landscape](./week01/docs/loss_landscape.jpg)

GD를 직관적으로 이해하기 위해 parameter space에서 정의되는 loss function을 생각해보자. 특정 parameter 벡터가 입력되면 loss 값이 출력으로 나오는 function 말이다. 특정 시점에서 parameter 입력이 주어졌을 때 GD를 통해서 우리는 loss가 최소화되는 지점에 다다르고 싶어한다. 모든 parameter space에 대한 탐색은 costy하고 in-tractable하기 때문에 iteration을 거듭하면서 더 나은 결과를 기대하는 것이 Gradient Descent(Back propagation)이다.

위의 그림은 2차원보다 큰 parameter space에서 [loss surface를 시각화하는 방법에 대해서 제시한 논문](https://arxiv.org/pdf/1712.09913)을 참고하여 구현한 것이다. (50, 50)은 네트워크의 원점으로 이 주변을 탐색하면서 얻어진 loss 값이 색깔에 따라서 표현되어 있음을 확인할 수 있다. '지형이 고른 형태를 보일수록 안정적인 학습이 가능할 것이다.'와 같이 loss surface의 시각화를 통해서 여러 정보를 얻을 수 있다. 아래는 이 그림을 3차원으로 나타낸 것으로 gradient(=학습 방향)이 loss가 감소하는 방향인 것을 실험적으로도 얻을 수 있다.(수학적인 방법이 궁금하다면 loss function을 스칼라 출력을 내는 벡터 함수라고 생각한 후에 이의 gradient vector을 생각해보기를!)

| ReLU                             | Tanh                             |
| -------------------------------- | -------------------------------- |
| ![image](./week01/docs/relu.png) | ![image](./week01/docs/tanh.png) |

_궁금증을 가질만한 질문들_

1. GD는 결정론적인가?: random seed 등의 값이 똑같다면 학습 과정의 randomness는 없다. 모든 데이터셋을 한 배치로 훈련시키기 때문이다. -> SGD(Stochastic+GD)이 더 효율적인 이유가 무엇일까?
2. SGD의 loss surface는?: SGD는 배치의 샘플링 과정을 통해서 통계적 안정성을 얻을 수 있다. 위의 상황처럼 GD의 loss surface의 경우와 다르게 SGD의 경우는 어떻게 해석해야 할까?
3. training dataset vs test dataset -> 서로 loss surface가 어떻게 다를까?

-   Related Resources
    -   [Loss Surface](https://arxiv.org/pdf/1712.09913)
    -   [Capsule nets](https://medium.com/ai³-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b)
    -   [How Embedding Works](https://github.com/colah/NLP-RNNs-Representations-Post/blob/master/index.md)
    -   [Gradient Vector](https://en.wikipedia.org/wiki/Gradient)
    -   [What is convolution](https://www.youtube.com/watch?v=KuXjwB4LzSA)
    -   [Visualization by cs231n](https://www.youtube.com/watch?v=ta5fdaqDT3M)

**Week02: 역사를 훑다**

**About HW**

difficult mnist by me!

About dataset

-   이미지 크기 (3x64x64)입니다.
-   3가지 색깔로 이루어진 펜으로 5가지 질감과 색이 다른 종이에 쓰인 숫자를 맞춰야 합니다.
-   숫자는 0-9로 구성되어 있습니다.
-   숫자의 위치는 랜덤하게 설정되었고, 숫자의 크기 도한 랜덤하게 설정되었습니다.

> win the "tung tung tung tung sahur"

<a target="_blank" href="https://colab.research.google.com/github/yoonhero/sig_archive/blob/main/hw/example_code.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
