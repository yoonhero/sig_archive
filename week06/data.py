import chess
import chess.pgn
import io
import tqdm
import numpy as np
import os
import multiprocessing
import sys

# Increase recursion limit to handle deep recursion during pickling
sys.setrecursionlimit(10000)

from state import State, visualize_action, decode_action, break_down_uci
from state import vectorize, OPPONENT_TOK, ME_TOK, PAD_TOK
import dtypes

username = "yoonhero"
path = "./data/DATABASE4U.pgn"
mode = "sequence" # or cnn
save_path = f"./data/processed/database4u_{mode}.npz"

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

def load_game_from_string(pgn: str):
    stream = io.StringIO(pgn)
    game = chess.pgn.read_game(stream)
    return game

def parse_game(game, mode="sequence", max_length=140) -> tuple[any, dtypes.Actions, str]:
    # Protocols                         Type               Unambiguity                   Human friendly
    # UCI(universal chess interface)    Only movement(DAG) None                          No(Engine)
    # SAN(standard algebraic notation)  Including Actions  Probably(Additional notation) Yes(PGN)
        # dtypes: x(capture), +(check), #(checkmate)
        # pawn capture -> exd5 / promotion -> e8=Q
    # game -> 1. e3 {[%clk 0:09:58.9]} 1... e5 {[%clk 0:09:55.2]} ... 1-0
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
            try:
                action = vectorize(uci)
            except KeyError:
                raise ValueError("Invalid move")
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

# Define process_single_game outside of any class to avoid pickling issues
def process_single_game(args):
    game, mode, max_length = args
    try:
        return parse_game(game, mode, max_length)
    except ValueError:
        return None

def load_games(path, mode, max_length=None, save_path=None):
    processed_games = 0
    invalid_games = 0
    vectors, actions, results = [], [], []
    if os.path.exists(save_path):
        print(f"--- Loaded cache from {save_path} ---")
        data = np.load(save_path)
        return data["vectors"], data["actions"], data["results"]
    
    pgn = open(path)
    games = []
    while True:
        try:
            game = chess.pgn.read_game(pgn)
        except UnicodeDecodeError:
            print("UnicodeDecodeError")
            print(len(games))
            break
        if game is not None:
            games.append(game)
        else: break
        if len(games) > 100000: break
    pgn.close()

    # Use a simpler approach with chunksize to reduce recursion depth
    tasks = [(g, mode, max_length) for g in games]
    with multiprocessing.Pool(processes=6) as pool:
        results_list = pool.map(process_single_game, tasks, chunksize=10)
        
    for result in results_list:
        if result is None:
            invalid_games += 1
            continue
        vectors.extend(result[0])
        actions.extend(result[1])
        results.extend(result[2])
        processed_games += 1
    
    vectors = np.stack(vectors, axis=0)
    actions = np.stack(actions, axis=0)
    results = np.vstack(results)
    assert vectors.shape[0] == actions.shape[0] == results.shape[0]
    print(f"Loaded {processed_games} games, {invalid_games} invalid games")
    if save_path:
        print(f"--- Saving to {save_path} ---")
        np.savez(save_path, vectors=vectors, actions=actions, results=results)
    return vectors, actions, results

if __name__ == "__main__":
    vectors, actions, results = load_games(path, mode, max_length=140, save_path=save_path)
    max_length = 73 # only evaluates the board.
    vectors = vectors[:1000, :max_length]
    results = results[:1000]
    embedding = 32
    batch_size = 512
    import torch
    from torch.utils.data import Dataset, DataLoader
    from state import total_tokens
    device = "cpu"
    class dataset(Dataset):
        def __init__(self, vectors, results):
            self.vectors = vectors
            self.results = results
        def __len__(self):
            return len(self.vectors)
        def __getitem__(self, idx):
            return torch.from_numpy(self.vectors[idx]).to(torch.long), torch.from_numpy(self.results[idx]).to(torch.float)
    ds = dataset(vectors, results)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    B, T = vectors.shape
    model = torch.nn.Sequential(
        torch.nn.Embedding(total_tokens, embedding),
        torch.nn.Flatten(1),
        torch.nn.Linear(max_length*embedding, 512),
        torch.nn.Tanh(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 256),
        torch.nn.Tanh(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(256, 1),
        torch.nn.Tanh(),
    ).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    for epoch in range(20):
        losses = 0
        acc = 0
        for _vectors, _results in tqdm.tqdm(dataloader):
            optimizer.zero_grad()
            _vectors = _vectors.to(device)
            _results = _results.to(device)
            output = model(_vectors)
            loss = criterion(output, _results)
            acc += (output.round() == _results).sum().item()
            loss.backward()
            optimizer.step()
            losses += loss.item()
        print(f"epoch {epoch+1}, Loss: {losses/len(dataloader):.4f}")
        print(f"acc: {acc/len(ds):.4f}")
    torch.save(model.state_dict(), f"./model/small.pth")

    # visualize_action(10)
    # uci = ["d7e8q", "a2b1r", "g7h8b", "h2g1q", "a2a1q"]
    # from state import encode_uci
    # for u in uci:
    #     print(f"START {u}")
    #     assert decode_action(encode_uci(u)) == u
    # # print(decode_action(encode_uci("d7qe8"))=="d7qe8")