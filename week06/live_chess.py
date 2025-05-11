import chess
import random
from flask import Flask, render_template, request
import sys
import matplotlib.pyplot as plt

from state import State
from data import embedding, n_layer, models
from data import AttentionPolicy
from agent import *

app = Flask(__name__)
    
state = State()
# agent = BasicSearchAgent(state, max_depth=3)
# model = AttentionPolicy(embedding=embedding, n_layer=n_layer)

model = AttentionPolicy(*models["medium"])
w = torch.load("./model/medium/100k_20.pth", weights_only=True)
w = {k: v for k, v in w.items() if "mask" not in k}
model.load_state_dict(w, strict=False) # terrible value network!
agent = TorchPolicyAgent(model, state, max_depth=6, max_leaf=6)
model.eval()
if sys.argv[-1] == "stockfish":
    opponent = StockfishAgent(state, skill_level=0)
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
    app.run(port=8080)