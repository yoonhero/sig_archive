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
2. 일반화는 어떻게 이루어지는거지?: test set의 비율을 조정하면서 실험해보기
3. 좌표계를 그렇게 왜곡시켰는데 파라메터 공간에 대해서 일반적인 GD를 쓸 수 있나?: Riemmanian metric+Natural Gradient

-   [Loss surface](./week01/loss_surface.ipynb) visualization of your model! (why residual connection is GOD.)
