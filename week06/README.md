### Make your chessbot

> live chess with you against you as human versus robot!

```bash
pip install -r requirements.txt
python live_chess.py # localhost:8080 for playing!!
# go to /?self=1 for bot self playing
# specify /?role="b" if you want to play as black else it's randomly chosen.
```

```
├── chess.com.py: 자신의 대국 데이터를 chess.com에서 다운로드하기.
├── data.py: pgn 데이터를 모델 학습할 수 있는 형식으로 바꾸기.(both supported CNN/Sequence Model in progress...)
    └── 기본적인 학습 logic 제공.
├── live_chess.py: 실시간으로 제작한 모델/봇을 웹상에서 테스트하기.
    └── BasicSearchAgent: 기본적인 evaluation을 가지고 minimax 알고리즘을 수행하는 예시
├── benchmark.py: 스톡피쉬와 겨루어서 현재 봇의 ELO를 추정하기.(in progress...)
├── requirements.txt: 필요한 패키지들
```

**해볼 만한 도전들**

-   Upgrade evaluation functions: 휴리스틱 함수들 or 딥러닝 모델으로
-   Search deep and thin: 더 깊은 탐색을 하기 위해서는 wide한 탐색을 하지 못한다. 어떤 선택지를 버릴지 휴리스틱하게/딥러닝 모델로 선택해보자.
-   Optimize search: 현재 minimax search 함수에서 다양한 cache를 통해 최적화! + alpha/beta pruning
    -   Algorithms: https://www.cs.cornell.edu/boom/2004sp/ProjectArch/Chess/algorithms.html#minmax
    -   Video: https://www.youtube.com/watch?v=_vqlIPDR2TU
-   Opening Book: 초창기에는 수읽기보다 이미 정해진 경로를 따라가는게 효과적이죠.
-   ...

my TODOs:

-   [x] Self play + refactoring
-   [ ] SPRT with stockfish or weeker opponents.
-   [ ] Minimax search tree visualization(evaluation comparison panel?)
-   [ ] Find a way to play 1/1 live chess contest
-   [ ] Lichess bot wrapper? or just SCSC discord play~.
