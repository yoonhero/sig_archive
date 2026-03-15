import chess
from chess import IllegalMoveError
import random
from typing import Optional
from flask import Flask, render_template, request
import time
import sys

from eval_example import evaluate_board
import dtypes
from state import State, decode_action, encode_uci

app = Flask(__name__)

class Agent():
    def __init__(self, state: State):
        self.state = state
        self.search_cache = {}

    def respond(self) -> Optional[dtypes.UCI]:
        if self.state.game_over(): return None
        self.state.detach()
        next_move = decode_action(self.predict())
        self.state.attach()
        self.state.push_uci(next_move) # try it empirically.
        self.state.detach()
        return next_move
    
    def predict(self) -> dtypes.Action:
        ...

    # get sorted actions by prob order in the current state
    def get_candidates(self) -> dtypes.Actions:
        ...
    # scoring the current state. -> You can replace this part with Neural networks!
    def evaluate_board(self, me: bool) -> float: 
        ...
    # learn more: https://en.wikipedia.org/wiki/Minimax
    def minimax_search(self, me: bool=True, max_leaf=3, cur_depth=0, max_depth=4):
        assert max_depth>0, "Max Depth must be greater than one."
        if cur_depth == max_depth or self.state.game_over(): # if we reach the maximum depth.
            return self.evaluate_board(me), 1
        candidates = self.get_candidates()
        values = []
        aggregation = 0
        for candidate in candidates:
            if len(values) == max_leaf: break
            self.state.push_uci(decode_action(candidate))
            v, n = self.minimax_search(me=not me, max_leaf=max_leaf, cur_depth=cur_depth+1, max_depth=max_depth)
            values.append((v, candidate))
            aggregation += n
            self.state.undo()
        values = sorted(values, key=lambda x: x[0], reverse=me) # define the current player's strategy

        if cur_depth == 0:
            print([(v, decode_action(c)) for (v, c) in values])
            return values[0][1], aggregation
        return values[0][0], aggregation

# Estimated ELO ranges for Stockfish skill levels
# These are approximate values based on community testing
SKILL_LEVEL_ELO_MAP = {
    0: (1200, 1450),    # Based on master-skill-0: 1320.1
    1: (1350, 1600),    # Based on master-skill-1: 1467.6
    2: (1500, 1750),    # Based on master-skill-2: 1608.4
    3: (1600, 1900),    # Based on master-skill-3: 1742.3
    4: (1800, 2050),    # Based on master-skill-4: 1922.9
    5: (2050, 2350),    # Based on master-skill-5: 2203.7
    6: (2250, 2500),    # Based on master-skill-6: 2363.2
    7: (2350, 2650),    # Based on master-skill-7: 2499.5
    8: (2450, 2750),    # Based on master-skill-8: 2596.2
    9: (2550, 2850),    # Based on master-skill-9: 2702.8
    10: (2650, 2950),   # Based on master-skill-10: 2788.3
    11: (2700, 3000),   # Based on master-skill-11: 2855.5
    12: (2800, 3050),   # Based on master-skill-12: 2923.1
    13: (2850, 3100),   # Based on master-skill-13: 2972.9
    14: (2900, 3150),   # Based on master-skill-14: 3024.8
    15: (2950, 3200),   # Based on master-skill-15: 3069.5
    16: (3000, 3250),   # Based on master-skill-16: 3111.2
    17: (3000, 3300),   # Based on master-skill-17: 3141.3
    18: (3050, 3300),   # Based on master-skill-18: 3170.3
    19: (3050, 3350),   # Based on master-skill-19: 3191.1
    20: (3300, 3600),   # Max strength, estimation beyond table
}

class StockfishAgent(Agent):
    def __init__(self, state: State, stockfish_path: str, skill_level: int = 0, timeout: float = 0.5):
        super().__init__(state)
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path, timeout=timeout)
        self.engine.configure({"Skill Level": skill_level})
        self.skill_level = skill_level
        print(f"Stockfish {skill_level}(={self.get_estimated_elo()}) is ready.")
    def predict(self) -> dtypes.UCI:
        return encode_uci(self.engine.play(self.state.board, chess.engine.Limit(time=0.1)).move.uci())
    def get_estimated_elo(self):
        return SKILL_LEVEL_ELO_MAP[self.skill_level]
    def __del__(self):
        self.engine.close()

class RandomAgent(Agent):
    def predict(self) -> str:
        return random.choice(self.state.get_legel_actions())

# You can build even powerful without complex DLs: following https://www.youtube.com/watch?v=U4ogK0MIzqk
class BasicSearchAgent(Agent):
    def __init__(self, state: State, max_depth=4, max_leaf=-1):
        super().__init__(state)
        self.max_depth = max_depth
        self.max_leaf = max_leaf
    def get_candidates(self, **kwargs) -> dtypes.Actions:
        legal_actions = self.state.get_legel_actions()
        return legal_actions
    def evaluate_board(self, me: bool) -> float:
        if self.state.game_over():
            return float("-inf") if me else float("inf")
        score = evaluate_board(self.state.__repr__())
        cur_turn = self.state.board.turn
        am_i_white = (cur_turn == chess.WHITE and me) or (cur_turn == chess.BLACK and not me)
        return score if am_i_white else -score
    def predict(self) -> dtypes.UCI:
        start = time.monotonic()
        next_uci, aggregation = self.minimax_search(max_leaf=self.max_leaf, max_depth=self.max_depth)
        duration = time.monotonic() - start
        print(f"Search {aggregation}items in {duration}s with {aggregation/duration/1000:.3f}k/s")
        return next_uci
    
class TorchSearchAgent(BasicSearchAgent):
    def __init__(self, state: State, max_depth=4, max_leaf=-1):
        super().__init__(state, max_depth, max_leaf)
        import torch
        from state import total_tokens
        self.model  = torch.nn.Sequential(
            torch.nn.Embedding(total_tokens, 16),
            torch.nn.Flatten(1),
            torch.nn.Linear(73*16, 512),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 1),
            torch.nn.Tanh(),
        )
        self.model.load_state_dict(torch.load("./simple_value_network.pth", weights_only=True)) # terrible value network!
        self.model.eval()
    def get_candidates(self) -> dtypes.Actions: # explore 1
        legal_actions = self.state.get_legel_actions()
        return legal_actions
    def evaluate_board(self, me: bool) -> float:
        if self.state.game_over():
            return float("-inf") if me else float("inf")
        import torch
        x = torch.Tensor(self.state.serialize("sequence")[:73]).unsqueeze(0).to(torch.long)
        score = self.model(x).item()
        cur_turn = self.state.board.turn
        am_i_white = (cur_turn == chess.WHITE and me) or (cur_turn == chess.BLACK and not me)
        return score if am_i_white else -score

state = State()
# agent = BasicSearchAgent(state, max_depth=3)
agent = TorchSearchAgent(state, max_depth=3, max_leaf=-1)
if sys.argv[-1] == "stockfish":
    opponent = StockfishAgent(state, stockfish_path="**", skill_level=0)
else:
    opponent = BasicSearchAgent(state, max_depth=3, max_leaf=-1)

@app.route("/")
def play_chess():
    state.reset()
    self_play = request.args.get("self")
    your_role_ = random.choice(["b", "w"]) if not self_play else "w"
    your_role = request.args.get("role", your_role_)
    return render_template("index.html", your_role=your_role, self_play=self_play)

@app.route("/move", methods=["PUT"])
def move_chesspiece():
    body = request.get_json()
    self_play = body["self_play"] == "1"
    history = body["history"]
    benchmark = body["benchmark"]
    if history and not self_play:
        prev_move = history[-1]
        uci = prev_move["from"]+prev_move["to"]
        if "promotion" in prev_move:
            uci += prev_move["promotion"]
        state.push_uci(uci)
    if benchmark:
        if state.board.turn == chess.WHITE:
            uci_move = agent.respond()
        else:
            uci_move = opponent.respond()
    else:
        uci_move = agent.respond()

    response = {}
    if uci_move is not None:
        response["move"] = state.current_move
    if state.game_over():
        response["game_result"] = state.game_result()
    return response

@app.route("/reset", methods=["PUT"])
def reset():
    state.reset()
    return {"ok": 200}

if __name__  == "__main__":
    app.run(port=8080, debug=True)