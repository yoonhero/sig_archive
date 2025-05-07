import chess
import chess.pgn
import io
import tqdm
import numpy as np
import os

from state import State, visualize_action, decode_action, break_down_uci
from state import vectorize, OPPONENT_TOK, ME_TOK, PAD_TOK
import dtypes

username = "yoonhero"
path = "./my_chess.pgn"
mode = "sequence" # or cnn
save_path = f"./data/processed/my_chess_{mode}.npz"

def prepare_sequence_data(vector, prev_action):
    if prev_action is not None:
        vector += [vectorize(OPPONENT_TOK), prev_action]
    vector += [vectorize(ME_TOK)]
    return vector

def pad_sequence_data(vector, max_length):
    vector = vector[:max_length]
    if len(vector) < max_length:
        vector = np.pad(vector, (0, max_length-len(vector)), mode="constant", constant_values=vectorize(PAD_TOK))
    return vector

def parse_game(pgn, mode="sequence", max_length=140) -> tuple[any, dtypes.Actions, str]:
    # Protocols                         Type               Unambiguity                   Human friendly
    # UCI(universal chess interface)    Only movement(DAG) None                          No(Engine)
    # SAN(standard algebraic notation)  Including Actions  Probably(Additional notation) Yes(PGN)
        # dtypes: x(capture), +(check), #(checkmate)
        # pawn capture -> exd5 / promotion -> e8=Q
    # game -> 1. e3 {[%clk 0:09:58.9]} 1... e5 {[%clk 0:09:55.2]} ... 1-0
    stream = io.StringIO(pgn)
    game = chess.pgn.read_game(stream)
    result = game.headers.get("Result")
    game_result = 1 if result == "1-0" else -1 if result == "0-1" else 0 # 1: white win / -1: black win / 0: draw
    board = game.board()
    vectors = []
    actions = []
    state = State(board)
    prev_action = None
    for move in game.mainline_moves():
        try:
            vector = state.serialize(mode) # sequence -> length / cnn -> 15 channels
        except AssertionError: # not a noraml game
            break
        uci = str(move)
        state.push(move)
        if mode == "sequence": # output next_action
            action = vectorize(uci)
            vector = prepare_sequence_data(vector, prev_action) + [action]
            vector = np.array(vector)
            vector = pad_sequence_data(vector, max_length)
            prev_action = action
            if action is None:
                raise ValueError("Invalid move")
        elif mode == "cnn": # output: 3, 8, 8 -> from, to, promotion
            action = np.zeros((3, 8, 8))
            from_row, from_col, to_row, to_col, promotion = break_down_uci(uci) # promotion is 1~4
            action[0, from_row, from_col] = 1
            action[1, to_row, to_col] = 1
            if promotion and promotion > 0:
                action[2, promotion-1, :] = 1 # nbrq
        vectors.append(vector)
        actions.append(action)
    else:
        if len(vectors) == 0:
            raise ValueError("No moves")
        return vectors, actions, [game_result] * len(vectors)
    raise ValueError("Not a normal game")

def load_games(path, mode, max_length=None, save_path=None):
    total_games = 0
    vectors, actions, results = [], [], []
    if os.path.exists(save_path):
        print(f"--- Loaded cache to {save_path} ---")
        data = np.load(save_path)
        return data["vectors"], data["actions"], data["results"]
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm.tqdm(f.read().rstrip().split("\n\n\n")):
            try:
                _vectors, _actions, _results = parse_game(line, mode, max_length)
                vectors.extend(_vectors)
                actions.extend(_actions)
                results.extend(_results)
                total_games += 1
            except ValueError:
                continue
    vectors = np.stack(vectors, axis=0)
    actions = np.stack(actions, axis=0)
    results = np.vstack(results)
    assert vectors.shape[0] == actions.shape[0] == results.shape[0]
    print(f"Loaded {total_games} games")
    if save_path:
        print(f"--- Saving to {save_path} ---")
        np.savez(save_path, vectors=vectors, actions=actions, results=results)
    return vectors, actions, results

if __name__ == "__main__":
    vectors, actions, results = load_games(path, mode, save_path=save_path)
    max_length = 73 # only evaluates the board.
    embedding = 64
    import torch
    from state import total_tokens
    print(f"DS: {total_tokens}")
    vectors = torch.from_numpy(vectors).to(torch.long)
    vectors = vectors[:, :max_length]
    results = torch.from_numpy(results).to(torch.float)
    B, T = vectors.shape
    batch_size = 128
    def sampler():
        while True:
            yield torch.randint(0, B, (batch_size,), dtype=torch.int)
    # Simple value network
    model = torch.nn.Sequential(
        torch.nn.Embedding(total_tokens, embedding),
        torch.nn.Flatten(1),
        torch.nn.Linear(max_length*embedding, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 1),
        torch.nn.Tanh()
    )
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    sample = sampler()
    for i in range(40):
        batch_indices = next(sample)
        _vectors = vectors[batch_indices]
        _results = results[batch_indices]
        optimizer.zero_grad()
        output = model(_vectors)
        loss = criterion(output, _results)
        loss.backward()
        optimizer.step()
        print(f"step {i+1}, Loss: {loss.item()}")
    torch.save(model.state_dict(), f"./simple_value_network_{mode}.pth")

    # visualize_action(10)
    # uci = ["d7e8q", "a2b1r", "g7h8b", "h2g1q", "a2a1q"]
    # from state import encode_uci
    # for u in uci:
    #     print(f"START {u}")
    #     assert decode_action(encode_uci(u)) == u
    # # print(decode_action(encode_uci("d7qe8"))=="d7qe8")