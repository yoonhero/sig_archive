## Interactive textbook

-   PPT 스타일을 그대로 디지털화한 Svelte 교재 앱: `cd interactive-textbook && npm install && npm run dev`
-   `mlsig.mmeme.org`/`nn-complete` toy project 경험은 링크 참고가 아니라 앱 내부 mini lesson/widget으로 흡수한다.
-   변환 계획과 챕터 맵: [BUILD_PLAN.md](BUILD_PLAN.md)
-   PDF/README 변환 소스맵: [textbook/SOURCE_MAP.md](textbook/SOURCE_MAP.md)
-   이전 정적 프로토타입: [textbook/index.html](textbook/index.html)
-   현재 Svelte 구현: [interactive-textbook](interactive-textbook)

## 1주차: \_\_init\_\_

-   명령형/선언형 프로그래밍: 인공지능=What에 대해 생각해보자.
-   학습 loop=loss 정의 -> backprop으로 gradient 계산 -> optimizer로 update / 구조=ad-hoc.
-   toy game: [해보자!](w1.mlsig.mmeme.org/) -> 생각보다 어려움!
-   과제: 4 element sorting을 만들어보자.

## 2주차: 간단한 수학(?)

-   과제 review: XOR/SORT를 다시 보면서 회로를 어떻게 짜는지 점검해보자.
-   예측과 정답의 차이: loss로 정의하고, backpropagation으로 전달해보자.
-   학습의 핵심: `parameter -= lr * gradient` 꼴로 반대로 이동해보자.
-   toy task 실습: 각 블록의 backward를 구현하고 학습해보자.
-   과제: 오차역전파로 [관악산을 등반해보자](https://mlsig.mmeme.org/w2/hard).

## 3주차: 이 노가다를 끝내러 왔다. (행렬)

-   우리가 하던 노가다: 데이터 주기, 회로 만들기, 학습하기의 흐름을 다시 묶어보자.
-   연결의 표현: 손으로 맞추던 회로를 행렬과 벡터로 써보자.
-   선형대수 관점: 곱하기, 더하기, ReLU를 행렬로 보면 좀 더 편해진다.
-   학습지 실습: AND/XOR를 예시로 순전파, loss, gradient, update를 손으로 해보자.
-   과제: `x ∈ R^4`에서 `sort(x)`, `median(x)`를 만들고 `dLoss/dx_i`를 구해보자.
