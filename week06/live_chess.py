import chess
from chess import IllegalMoveError
import random
from typing import TypeAlias, Optional
from flask import Flask, render_template, request, jsonify

from data import encode_uci, decode_action, tokenize_fen
from eval_example import evaluate_board
import dtypes

app = Flask(__name__)

class State():
    def __init__(self, board=None):
        self.board = chess.Board()
        if board is not None:
            self.board = board
        self.cur: Optional[dtypes.SAN] = None
    def __repr__(self) -> str:
        return ["".join(["." * int(s) if s.isdigit() else s for s in line]) for line in state.board.board_fen().split("/")  ]
    def serialize(self) -> dtypes.Tokens:
        return tokenize_fen(self.board.board_fen())
    def push(self, move: chess.Move):
        self.cur = self.board.san(move)
        self.board.push(move)
    def push_uci(self, uci: dtypes.UCI):
        move = chess.Move.from_uci(uci)
        self.push(move)
    def current_move(self) -> dtypes.SAN:
        return self.cur
    def get_legel_actions(self) -> dtypes.Actions:
        legal_moves = [
            encode_uci(move.uci())
            for move in self.board.legal_moves
        ]
        return legal_moves
    def reset(self):
        self.board.reset()
        self.cur = None
    def copy(self): return self.board.copy()
    def clone(self): return State(self.copy())
    def __str__(self):
        return str(self.board)
    def game_over(self) -> bool:
        return self.board.is_game_over()
    def game_result(self) -> str:
        return self.board.outcome().result()

class Agent():
    def __init__(self, state: State):
        self.state = state
        self.search_cache = {}

    def respond(self) -> Optional[dtypes.UCI]:
        if self.state.game_over(): return None
        while True:
            next_move = decode_action(self.predict())
            try:
                self.state.push_uci(next_move) # try it empirically.
                break
            except IllegalMoveError:
                continue
        print(self.state)
        return next_move
    
    def predict(self) -> dtypes.Action:
        ...
    # get sorted by prob of move in the current state
    def get_candidates(self, state: State) -> dtypes.Actions:
        ...
    # scoring the current state.
    def evaluate_board(self, state: State, me: bool) -> float: 
        ...
    # learn more: https://en.wikipedia.org/wiki/Minimax
    def minimax_search(self, state: State=None, me: bool=True, max_leaf=3, cur_depth=0, max_depth=4):
        assert max_depth>0, "Max Depth must be greater than one."
        if state is None:
            state = self.state.clone()
        if cur_depth == max_depth or state.game_over(): # if we reach the maximum depth.
            return self.evaluate_board(state, me)
        candidates = self.get_candidates(state)
        states = []
        for candidate in candidates:
            if max_leaf != -1 and len(states) == max_leaf: break
            try:
                (new_state:=state.clone()).push_uci(decode_action(candidate))
                states.append(new_state)
            except IllegalMoveError: # generated move can be probably illegal move!
                pass
        # is_maximizing_player = cur_depth % 2 == 0 
        play_strategy = max if me else min # define the current player's strategy
        values = [self.minimax_search(state, me=not me, max_leaf=max_leaf, cur_depth=cur_depth+1, max_depth=max_depth) for state in states]
        if len(values) == 0:
            value = float("-inf") if me else float("inf")
        else:
            value = play_strategy(values)
        if cur_depth == 0:
            print(value)
            return candidates[values.index(value)]
        return value

class RandomAgent(Agent):
    def predict(self) -> str:
        return random.choice(self.state.get_legel_actions())

class BasicSearchAgent(Agent):
    def get_candidates(self, state: State) -> dtypes.Actions:
        legal_actions = state.get_legel_actions()
        return legal_actions
    def evaluate_board(self, state: State, me: bool) -> float:
        if state.game_over():
            return float("-inf") if me else float("inf")
        score = evaluate_board(state.__repr__())
        am_i_white = (state.board.turn == chess.WHITE and me) or (state.board.turn == chess.BLACK and not me)
        return score if am_i_white else -score
    def predict(self) -> dtypes.UCI:
        return self.minimax_search(max_leaf=-1, max_depth=3)

state = State()
print(state.serialize())
# agent = RandomAgent(state)
agent = BasicSearchAgent(state)

@app.route("/")
def play_chess():
    state.reset()
    self_play = request.args.get("self")
    your_role = random.choice(["b", "w"]) if not self_play else "w"
    return render_template("index.html", your_role=your_role, self_play=self_play)

@app.route("/move", methods=["PUT"])
def move_chesspiece():
    body = request.get_json()
    self_play = body["self_play"] == "1"
    history = body["history"]
    if history and not self_play:
        prev_move = history[-1]
        uci = prev_move["from"]+prev_move["to"]
        if "promotion" in prev_move:
            uci += prev_move["promotion"]
        state.push_uci(uci)
    uci_move = agent.respond()

    response = {}
    if uci_move is not None:
        response["move"] = state.current_move()
    if state.game_over():
        response["game_result"] = state.game_result()
    return response

@app.route("/reset", methods=["PUT"])
def reset():
    state.reset()
    return {"ok": 200}

if __name__  == "__main__":
    app.run(debug=True, port=8080)