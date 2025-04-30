import chess
import random
from chess import IllegalMoveError, InvalidMoveError
from flask import Flask, render_template, request, jsonify

from data import encode_uci, decode_action, tokenize_fen

app = Flask(__name__)

class State():
    def __init__(self):
        self.board = chess.Board()
        self.cur = None
    def reset(self):
        self.board.reset()
        self.cur = None
    def push(self, move: chess.Move):
        self.cur = self.board.san(move)
        self.board.push(move)
    def push_uci(self, uci: str):
        move = chess.Move.from_uci(uci)
        self.push(move)
    def current_move(self):
        return self.cur
    def serialize(self):
        return tokenize_fen(self.board.board_fen())
    def get_legal_moves(self):
        legal_moves = [
            encode_uci(move.uci())
            for move in self.board.legal_moves
        ]
        return legal_moves
    def __str__(self):
        return str(self.board)
    
class Agent():
    def __init__(self, state: State):
        self.state = state

    def respond(self):
        while True:
            next_move = decode_action(self.predict())
            try:
                self.state.push_uci(next_move) # try it empirically.
                break
            except IllegalMoveError:
                continue
        print(self.state)
        return next_move
    
    def predict(self) -> str:
        return NotImplementedError

class RandomAgent(Agent):
    def predict(self):
        return random.choice(self.state.get_legal_moves())

state = State()
print(state.serialize())
agent = RandomAgent(state)

@app.route("/")
def play_chess():
    state.reset()
    return render_template("index.html")

@app.route("/move", methods=["PUT"])
def move_chesspiece():
    body = request.get_json()
    user_move = (history:=body["history"])[-1]
    uci = user_move["from"]+user_move["to"]
    if "promotion" in user_move.keys():
        uci += user_move["promotion"]
    state.push_uci(uci)
    _ = agent.respond()
    return {"move": state.current_move()}

if __name__  == "__main__":
    app.run(debug=True, port=8080)