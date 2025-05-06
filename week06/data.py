import chess
import chess.pgn
import io
import tqdm
import numpy as np

from state import State, encode_uci, visualize_action, decode_action
import dtypes

username = "yoonhero"
path = "./my_chess.pgn"
mode = "sequence" # or cnn

def parse_game(pgn, mode="sequence") -> tuple[any, dtypes.Actions, str]:
    # Protocols                         Type               Unambiguity                   Human friendly
    # UCI(universal chess interface)    Only movement(DAG) None                          No(Engine)
    # SAN(standard algebraic notation)  Including Actions  Probably(Additional notation) Yes(PGN)
        # dtypes: x(capture), +(check), #(checkmate)
        # pawn capture -> exd5 / promotion -> e8=Q
    # game -> 1. e3 {[%clk 0:09:58.9]} 1... e5 {[%clk 0:09:55.2]} ... 1-0
    stream = io.StringIO(pgn)
    game = chess.pgn.read_game(stream)
    game_result = game.headers.get("Result")
    board = game.board()
    fens = []
    actions = []
    for move in game.mainline_moves():
        state = State(board)
        arr = state.serialize(mode)
        # fens.append(board.board_fen())
        board.push(move)
        action = encode_uci(str(move))
        if action is None:
            return [], [], None
        actions.append(action)

    return fens, actions, game_result

def load_games(path, mode):
    games = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm.tqdm(f.read().rstrip().split("\n\n\n")[:10]):
            games.append(parse_game(line))
        
    return games

if __name__ == "__main__":
    data = load_games(path, mode)
    visualize_action(10)
    uci = ["d7e8q", "a2b1r", "g7h8b", "h2g1q", "a2a1q"]
    for u in uci:
        print(f"START {u}")
        assert decode_action(encode_uci(u)) == u
    # print(decode_action(encode_uci("d7qe8"))=="d7qe8")