## Sem2 Week01

- 정보가 무엇인지 직관적인 예시로 이해하고, AI에서 정보 개념이 왜 중요한지 살펴봅니다.
- Shannon 관점의 언어 모델, n-gram 실험, scaling law로 이어지는 흐름을 연결합니다.
- 정수 기반 토크나이징을 그대로 믿기 어려운 이유도 함께 다룹니다.

## Sem2 Week02

- `nalgae.txt`를 문자 단위로 토크나이즈해 n-gram 학습용 데이터와 분포 시각화를 만듭니다.
- 간단한 다음 글자 예측 실험으로 엔트로피, 생성, softmax 출력을 직접 확인합니다.
- 작은 예제로 오차 역전파와 gradient 계산을 numpy/torch 수준에서 따라갑니다.

## Sem2 Week03

- 문자 단위 n-gram 모델을 PyTorch로 옮겨 autograd와 computation graph 동작을 확인합니다.
- `torchviz`, profiler, custom autograd 함수로 forward/backward 구조를 뜯어봅니다.
- 임베딩 기반 모델을 실제로 학습시키며 텍스트 생성까지 이어집니다.

## Sem2 Week04

- 정수 덧셈·뺄셈 식 데이터를 만들어 문자 단위 RNN이 결과를 생성하도록 학습합니다.
- `model.py`, `utils.py`, `adder.pth`로 TinyRNN 구조, 토크나이저, 학습 결과를 분리해 둡니다.
- hidden state, padding, 샘플링, MPS 실행 특성까지 함께 실험합니다.

## Sem2 Week05

- 사칙연산 데이터셋을 확장하고, RNN 위에 causal self-attention을 얹은 ALU 형태 모델을 실험합니다.
- `=` 이후 정답 구간만 학습하도록 mask를 두어 연산 결과 생성에 집중시킵니다.
- attention, residual, dropout, 초기화 같은 transformer 계열 요소를 작은 모델에 직접 붙여봅니다.
