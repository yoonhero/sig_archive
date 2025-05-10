import chess
import chess.pgn
import io
import tqdm
import numpy as np
import os
import multiprocessing
import sys
from PIL import Image
import matplotlib.pyplot as plt

# Increase recursion limit to handle deep recursion during pickling
sys.setrecursionlimit(10000)

from state import State, visualize_action, decode_action, break_down_uci
from state import vectorize, OPPONENT_TOK, ME_TOK, PAD_TOK, total_tokens
import dtypes

def prepare_sequence_data(vector, prev_action):
    if prev_action is not None:
        vector += [vectorize(OPPONENT_TOK), prev_action]
    vector += [vectorize(ME_TOK)]
    return vector

def pad_sequence_data(vector, max_length):
    if len(vector) > max_length:
        print(f"EXCEEDS MAX LENGTH with {len(vector)}")
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
            vector = prepare_sequence_data(vector, prev_action) + [total_tokens + bool(state.board.turn)] + [action]
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
        if len(games) > 20000: break
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

board_length = 73 # only evaluates the board.
n_layer = 5
embedding = 32

from torch.utils.data import Dataset, DataLoader
class dataset(Dataset):
    def __init__(self, vectors, results):
        self.vectors = vectors
        self.results = results
    def __len__(self):
        return len(self.vectors)
    def __getitem__(self, idx):
        return torch.from_numpy(self.vectors[idx]).to(torch.long), torch.from_numpy(self.results[idx]).to(torch.float)

def make_dataloader(vectors, results):
    import sklearn.model_selection
    train_vectors, test_vectors, train_results, test_results = sklearn.model_selection.train_test_split(vectors, results, test_size=0.1, random_state=42)
    ds = dataset(train_vectors, train_results)
    test_ds = dataset(test_vectors, test_results)
    batch_size = 512
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return dataloader, test_dataloader

import torch.nn
class AttentionBlock(torch.nn.Module):
    def __init__(self, demb, nth_layer):
        super().__init__()
        self.act = torch.nn.ReLU()
        self.norm = torch.nn.LayerNorm(demb)
        self.qkv = torch.nn.Linear(demb, demb*3)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(demb, demb*3),
            torch.nn.LayerNorm(demb*3),
            torch.nn.ReLU(),
            torch.nn.Linear(demb*3, demb),
        )
        mask = torch.tril(torch.ones(200, 200)).view(1, 200, 200)
        mask[:, :73, :73] = 1 # mask sure they can view all board
        self.register_buffer("mask", mask.bool())
        self.nth_layer = nth_layer

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        x = self.qkv(self.act(self.norm((start:=x))))
        q, k, v = x.split(C, dim=2)
        attn = q @ k.transpose(-2, -1) * (1 / np.sqrt(C))
        attn = attn.masked_fill(~self.mask[:,:T,:T], float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        # plt.imshow(attn[0].detach().cpu().view(140, 140).numpy())
        # plt.savefig(f"./model/mask_{self.nth_layer}.png")
        out = attn @ v
        x = self.mlp(out)
        return x + start

class AttentionV(torch.nn.Module):
    def __init__(self, embedding, n_layer, max_length):
        super().__init__()
        self.embedding = torch.nn.Embedding(total_tokens, embedding)
        self.blocks = torch.nn.ModuleList([AttentionBlock(embedding, i) for i in range(n_layer)])
        self.norm = torch.nn.LayerNorm(max_length*embedding)
        self.linear = torch.nn.Linear(max_length*embedding, 1)
        self.act = torch.nn.Tanh()
    def forward(self, x):
        B = x.size(0)
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = x.reshape(B, -1)
        x = self.act(self.norm(x))
        x = self.act(self.linear(x))
        return x

class AttentionPolicy(torch.nn.Module):
    def __init__(self, embedding, n_layer, with_turn=False):
        super().__init__()
        if with_turn:
            tokens = total_tokens + 2
        else: tokens = total_tokens
        self.emb = torch.nn.Embedding(tokens, embedding)
        self.blocks = torch.nn.ModuleList([AttentionBlock(embedding, i) for i in range(n_layer)])
        self.norm = torch.nn.LayerNorm(embedding)
        self.linear = torch.nn.Linear(embedding, tokens)
        self.act = torch.nn.ReLU()
    def forward(self, x, train=False):
        x = self.emb((start:=x))
        for block in self.blocks:
            x = block(x)
        x = self.act(self.norm(x))
        logits = self.linear(x)
        if train:
            return torch.nn.functional.cross_entropy(logits[:, board_length:-1].contiguous().view(-1, total_tokens+2), start[:, board_length+1:].contiguous().view(-1), ignore_index=vectorize(PAD_TOK), reduction="mean")
        return logits

if __name__ == "__main__":
    username = "yoonhero"
    path = "./data/DATABASE4U.pgn"
    mode = "sequence" # or cnn
    save_path = f"./data/processed/database4u_withturn_{mode}.npz"
    vectors, actions, results = load_games(path, mode, max_length=160, save_path=save_path)
    print(vectors.shape, actions.shape, results.shape)
    # vectors = vectors[:100000]
    # results = results[:100000]
    device = "mps"
    dataloader, test_dataloader = make_dataloader(vectors, results)

    # model = AttentionPolicy(32, 5).to(device) -> small
    # model = AttentionPolicy(32, 2, True).to(device) -> tiny
    # model = AttentionPolicy(64, 10, True).to(device) -> medium
    model = AttentionPolicy(128, 12, True).to(device)

    print(sum([p.nelement() for p in model.parameters()]))
    # criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    epochs = 10
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        loss_ = 0
        acc = 0
        for _vectors, _results in tqdm.tqdm(dataloader):
            optimizer.zero_grad()
            _vectors = _vectors.to(device)
            # _results = _results.to(device)
            loss = model(_vectors, True)
            # loss = criterion(output, _results)
            loss.backward()
            optimizer.step()
            loss_ += loss.item()
        print(f"epoch {epoch+1}, Loss: {loss_/len(dataloader):.4f}")
        test_loss_ = 0
        with torch.no_grad():
            model.eval()
            for _vectors, _results in tqdm.tqdm(test_dataloader):
                _vectors = _vectors.to(device)
                # _results = _results.to(device)
                loss = model(_vectors, True)
                # loss = criterion(output, _results)
                test_loss_ += loss.item()
            model.train()
        print(f"epoch {epoch+1}, Test Loss: {test_loss_/len(test_dataloader):.4f}")
        train_losses.append(loss_/len(dataloader))
        test_losses.append(test_loss_/len(test_dataloader))
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f"./model/tiny_{epoch+1}.pth")

    # torch.save(model.state_dict(), f"./model/tiny_{epoch+1}.pth")
    import matplotlib.pyplot as plt
    plt.plot(range(epochs), train_losses, label="train")
    plt.plot(range(epochs), test_losses, label="test")
    plt.savefig(f"./model/medium_{epochs}.png")
    plt.legend()
    plt.show()
    
    # visualize_action(10)
    # uci = ["d7e8q", "a2b1r", "g7h8b", "h2g1q", "a2a1q"]
    # from state import encode_uci
    # for u in uci:
    #     print(f"START {u}")
    #     assert decode_action(encode_uci(u)) == u
    # # print(decode_action(encode_uci("d7qe8"))=="d7qe8")