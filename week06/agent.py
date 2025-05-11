import chess
import random
from typing import Optional
import time
import torch
import matplotlib.pyplot as plt

from eval_example import evaluate_board
import dtypes
from state import State, decode_action
from state import encode_uci
from benchmark import SKILL_LEVEL_ELO_MAP
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
    def minimax_search(self, me: bool=True, alpha=float("-inf"), beta=float("inf"), max_leaf=3, cur_depth=0, max_depth=4):
        assert max_depth>0, "Max Depth must be greater than one."
        if cur_depth == max_depth or self.state.game_over(): # if we reach the maximum depth.
            return self.evaluate_board(me), 1
        candidates = self.get_candidates()
        values = []
        aggregation = 0
        for candidate in candidates:
            if len(values) == max_leaf: break
            self.state.push_uci(decode_action(candidate))
            v, n = self.minimax_search(me=not me, alpha=alpha, beta=beta, max_leaf=max_leaf, cur_depth=cur_depth+1, max_depth=max_depth)
            values.append((v, candidate))
            aggregation += n
            self.state.undo()
            if me: # alpha = 내 입장에서 최대치
                alpha = max(alpha, v)
                if alpha >= beta: # 내 입장에서 최대치가 상대방 입장에서 최소치보다 크면 상대방은 이걸 선택해주지 않는다!
                    break
            else: # beta = 상대방 입장에서 최소치
                beta = min(beta, v)
                if beta <= alpha:
                    break
        values = sorted(values, key=lambda x: x[0], reverse=me) # define the current player's strategy

        if cur_depth == 0:
            print([(v, decode_action(c)) for (v, c) in values])
            # print([decode_action(action) for action in action_by_probs[:5]])
            return values[0][1], aggregation
        return values[0][0], aggregation

class StockfishAgent(Agent):
    def __init__(self, state: State, skill_level: int = 0):
        super().__init__(state)
        self.engine = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish", timeout=2)
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
    def __init__(self, model, state: State, max_depth=4, max_leaf=-1):
        super().__init__(state, max_depth, max_leaf)
        self.model = model
    def get_candidates(self) -> dtypes.Actions: # explore 1
        legal_actions = self.state.get_legel_actions()
        return legal_actions
    def evaluate_board(self, me: bool) -> float:
        if self.state.game_over():
            return float("-inf") if me else float("inf")
        x = torch.Tensor(self.state.serialize("sequence")[:73]).unsqueeze(0).to(torch.long)
        score = self.model(x).item()
        cur_turn = self.state.board.turn
        am_i_white = (cur_turn == chess.WHITE and me) or (cur_turn == chess.BLACK and not me)
        return score if am_i_white else -score

class TorchPolicyAgent(Agent):
    def __init__(self, model: torch.nn.Module, state: State, **kwargs):
        super().__init__(state)
        self.model = model
    def predict(self) -> dtypes.UCI:
        legal_actions = self.state.get_legel_actions()
        prev_action = encode_uci(self.state.board.move_stack[-1].uci()) if self.state.board.move_stack else None
        x = torch.Tensor(self.state.serialize("sequence", prev_action=prev_action)).unsqueeze(0).to(torch.long)
        logits = self.model(x)
        logits = logits[:, -1, :][0]
        conf = logits.softmax(dim=0)[legal_actions].sum()
        logits = logits[legal_actions]
        topk_logits, topk_indices = torch.topk(logits, 5, dim=0)
        probs = torch.nn.functional.softmax(topk_logits, dim=0)
        # fig, ax = plt.subplots(figsize=(10, 10))
        # ax.bar([decode_action(a) for a in legal_actions], probs.tolist())
        # fig.savefig("action_by_probs.png")
        H = (-probs.log() * probs).sum()
        print(H)
        index = topk_indices[torch.multinomial(probs, 1)].item()
        print(sorted({decode_action(a): p for a, p in zip(legal_actions, probs.tolist())}.items(), key=lambda x: x[1], reverse=True))
        print(legal_actions[index])
        return legal_actions[index]

class TorchPolicySearchAgent(BasicSearchAgent):
    def __init__(self, model: torch.nn.Module, state: State, max_depth=4, max_leaf=-1):
        super().__init__(state, max_depth, max_leaf)
        self.model = model
    @torch.no_grad()
    def get_candidates(self) -> dtypes.Actions:
        legal_actions = self.state.get_legel_actions()
        prev_action = encode_uci(self.state.board.move_stack[-1].uci()) if self.state.board.move_stack else None
        x = torch.Tensor(self.state.serialize("sequence", prev_action=prev_action)).unsqueeze(0).to(torch.long)
        logits = self.model(x)
        logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=1)
        probs = probs.cpu().numpy()
        actions = []
        action_by_probs = probs.argsort(axis=1)[0][::-1].tolist()
        while True:
            if len(action_by_probs) == 0:
                break
            action = action_by_probs.pop(0)
            if action in legal_actions:
                actions.append(action)
            if len(actions) == self.max_leaf:
                break
        return actions