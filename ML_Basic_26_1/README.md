# ML 기초 시그 26-1

머신러닝을 "마법"이 아니라 계산 그래프, 역전파, 데이터 구조, 하드웨어 제약으로 이해해보는 기초 시그입니다. 1-7주차는 손으로 만든 회로에서 출발해 행렬화, CNN/inductive bias, 시각화, GPU 최적화, 그리고 7주차 확장 발표까지 진행했습니다.

## 1주차: `__init__`

- 명령형/선언형 프로그래밍을 비교하며 인공지능이 무엇을 "하는지"부터 생각했다.
- 학습 loop를 `loss 정의 -> backprop으로 gradient 계산 -> optimizer update`로 잡았다.
- toy game으로 직접 회로를 만들어보고, 4-element sorting을 과제로 다뤘다.

## 2주차: 간단한 수학(?)

- XOR/SORT 과제를 리뷰하며 손으로 만든 회로의 구조를 점검했다.
- 예측과 정답의 차이를 loss로 정의하고, gradient descent로 파라미터를 움직이는 흐름을 배웠다.
- 블록별 backward를 구현하며 오차역전파가 정보를 어떻게 전달하는지 실습했다.
- 과제: 오차역전파로 관악산 등반하기.

## 3주차: 이 노가다를 끝내러 왔다. (행렬)

- 손으로 연결하던 회로를 행렬과 벡터로 표현했다.
- 선형 변환, bias, ReLU, loss, update를 계산 그래프와 선형대수 관점에서 정리했다.
- AND/XOR 학습지로 순전파, gradient, parameter update를 직접 계산했다.
- 과제: `x in R^4`에서 `sort(x)`, `median(x)`를 만들고 gradient를 해석하기.

## 4주차: 데이터, 어떻게 생겼니

- 깊은 행렬 네트워크만으로는 데이터의 대칭성과 구조를 충분히 살리기 어렵다는 문제를 다뤘다.
- 이미지의 translation equivariance/invariance를 예시로 convolution과 pooling의 의미를 설명했다.
- local receptive field, shared weight, group convolution, D4 symmetry 등 inductive bias의 아이디어를 소개했다.
- 과제: 상하/좌우가 이어진 이미지 구조에 맞게 CNN을 수정하는 아이디어 스케치.

## 5주차: 보고, 듣고, 맛보고, 질문하자

- 4주차 과제의 해답으로 circular padding, periodic CNN, D4 변환 등을 살펴봤다.
- loss만 보지 말고 예측값, weight/kernel, activation, feature space, saliency map, adversarial example을 시각화하는 이유를 정리했다.
- CNN 필터, Gabor filter, activation saturation, DeepDream/SmoothGrad/IG 등 모델 내부를 보는 관점을 소개했다.
- 과제: 주어진 convolution 필터와 네트워크 구조가 어떤 task를 수행하는지 역추론하는 "시각화 고고학".

## 6주차: Introduction to GPU Optimization

- 5주차 과제 해답으로 circular convolution과 DFT 관점을 정리했다.
- CPU/GPU 구조, thread-block-grid, register/shared/global memory 계층을 훑었다.
- roofline model로 compute-bound와 memory-bound를 구분하고 병목을 보는 법을 배웠다.
- 실습: CUDA vector add, matmul, tiling, kernel fusion, Triton으로 PyTorch eager execution의 메모리 왕복 줄이기.

## 7주차: 확장 발표

- **대학원생 하강법 vs 프론티어랩 하강법**: magic number, scaling law, hyperparameter, representation 관점에서 모델 개발의 감각을 다뤘다.
- **Everything Is A Graph**: grid/CNN, Transformer attention, 시계열과 음성을 graph/message passing 관점으로 바라봤다.
- **뇌는 역전파를 하지 않는다**: weight transport problem, two-phase problem, feedback alignment 등 backprop 대안과 한계를 소개했다.

## 실습 자료

- `conv/`: 3x3 kernel playground
- `convpuzzle/`: 11x11 convolution puzzle
- `interactive-textbook/`: 인터랙티브 교재
- `6주차실습/`: roofline, matmul, kernel fusion/Triton 노트북
