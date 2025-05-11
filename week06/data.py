#! /usr/bin/env python
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
import sys
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader

# Increase recursion limit to handle deep recursion during pickling
sys.setrecursionlimit(10000)

from models import *
from state import State, visualize_action, decode_action, break_down_uci
from state import vectorize, PAD_TOK
import dtypes

from essential import Logger, ONLINE

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
    attention_masks = []
    state = State(board)
    prev_action = None
    for move in game.mainline_moves():
        try:
            vector = state.serialize(mode, prev_action) # sequence -> length / cnn -> 15 channels
        except AssertionError: # not a noraml game
            break
        uci = str(move)
        state.push(move)
        attention_mask = None
        if mode == "sequence": # output next_action
            try:
                action = vectorize(uci)
            except KeyError:
                raise ValueError("Invalid move")
            vector += [action]
            action_index = len(vector) - 1
            vector = np.array(vector)
            vector = pad_sequence_data(vector, max_length)
            attention_mask = action_index
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
        if attention_mask is not None:
            attention_masks.append(attention_mask)
    else:
        if len(vectors) == 0:
            raise ValueError("No moves")
        return vectors, actions, attention_masks, [game_result] * len(vectors)
    raise ValueError("Not a normal game")

# Define process_single_game outside of any class to avoid pickling issues
def process_single_game(args):
    game, mode, max_length = args
    try:
        return parse_game(game, mode, max_length)
    except ValueError:
        return None

def load_games(path, mode, max_length=None, save_path=None, force_reload=False):
    processed_games = 0
    invalid_games = 0
    vectors, actions, attention_masks, results = [], [], [], []
    if os.path.exists(save_path) and not force_reload:
        print(f"--- Loaded cache from {save_path} ---")
        data = np.load(save_path)
        return data["vectors"], data["actions"], data["attention_masks"], data["results"]
    
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
        # if len(games) > 20000: break
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
        attention_masks.extend(result[2])
        results.extend(result[3])
        processed_games += 1
        if processed_games == 0:
            print(State.deserialize(result[0][0]))
    
    vectors = np.stack(vectors, axis=0)
    actions = np.stack(actions, axis=0)
    attention_masks = np.vstack(attention_masks)
    results = np.vstack(results)
    assert vectors.shape[0] == actions.shape[0] == results.shape[0]
    print(f"Loaded {processed_games} games, {invalid_games} invalid games")
    if save_path:
        print(f"--- Saving to {save_path} ---")
        np.savez(save_path, vectors=vectors, actions=actions, attention_masks=attention_masks, results=results)
    return vectors, actions, attention_masks, results

class dataset(Dataset):
    def __init__(self, vectors, results, attention_masks):
        self.vectors = vectors
        self.results = results
        self.attention_masks = attention_masks
    def __len__(self):
        return len(self.vectors)
    def __getitem__(self, idx):
        if len(self.attention_masks[idx]) == 0:
            return torch.from_numpy(self.vectors[idx]).to(torch.long), torch.from_numpy(self.results[idx]).to(torch.float), None
        else:
            return torch.from_numpy(self.vectors[idx]).to(torch.long), torch.from_numpy(self.results[idx]).to(torch.float), torch.from_numpy(self.attention_masks[idx]).to(torch.long)

def make_dataloader(vectors, results, attention_masks):
    import sklearn.model_selection
    train_vectors, test_vectors, train_results, test_results, train_attention_masks, test_attention_masks = sklearn.model_selection.train_test_split(vectors, results, attention_masks, test_size=0.1, random_state=42)
    ds = dataset(train_vectors, train_results, train_attention_masks)
    test_ds = dataset(test_vectors, test_results, test_attention_masks)
    batch_size = 512
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return dataloader, test_dataloader

# Policy Network Capcity by model size.
# 5/12/2025
# MODEL SIZE     Size           Min perplexity(1M boards)
# small          327914=0.3M    ~4=54??
# medium         932810=0.9M    ~2.9=18.17??(terrible!)
# large          2897354=2.9M   
models = {"tiny": (32, 2), "small": (32, 5), "medium": (64, 10), "large": (128, 12)}

if __name__ == "__main__":
    username = "yoonhero"
    path = "./data/DATABASE4U.pgn"
    mode = "sequence" # or cnn
    save_path = f"./data/processed/database4u_withturn_{mode}.npz"
    vectors, actions, attention_masks, results = load_games(path, mode, max_length=160, save_path=save_path)
    print(vectors.shape, actions.shape, attention_masks.shape, results.shape)
    dataset_samples_by_size = {
        "tiny": 10000,
        "small": 100000,
        "medium": 1000000,
        "large": 5900000,
    }
    num_of_samples = dataset_samples_by_size[os.getenv("DS", "small")]
    dataloader, test_dataloader = make_dataloader(vectors[:num_of_samples], results[:num_of_samples], attention_masks[:num_of_samples])
    
    device = "mps"
   
    model_size = os.getenv("MS", "small")
    model = AttentionPolicy(*models[model_size]).to(device)
    save_to = "./model/{model_size}".format(model_size=model_size)
    os.makedirs(save_to, exist_ok=True)
    print(f"{sum([p.nelement() for p in model.parameters()])} parameters")
    exit()

    run = Logger(ONLINE, run_name=f"{model_size}_{(k:=num_of_samples/1000):.0f}k", configs={"model_size": model_size, "num_of_samples": num_of_samples}, only_log=True, project="chess", settings=True)

    # criterion = torch.nn.MSELoss(
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    epochs = 20
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        loss = 0
        acc = 0
        for _vectors, _results, _attention_masks in tqdm.tqdm(dataloader):
            optimizer.zero_grad()
            _vectors = _vectors.to(device)
            # _results = _results.to(device)
            _loss = model(_vectors, True, _attention_masks)
            # loss = criterion(output, _results)
            _loss.backward()
            optimizer.step()
            loss += _loss.item()
        print(f"epoch {epoch+1}, Loss: {(loss:=loss/len(dataloader)):.4f}")
        test_loss = 0
        with torch.no_grad():
            model.eval()
            for _vectors, _results, _attention_masks in tqdm.tqdm(test_dataloader):
                _vectors = _vectors.to(device)
                _loss = model(_vectors, True, _attention_masks)
                test_loss += _loss.item()
            model.train()
        print(f"epoch {epoch+1}, Test Loss: {(test_loss:=test_loss/len(test_dataloader)):.4f}")
        train_losses.append(loss)
        test_losses.append(test_loss)
        run.log({"train/loss": loss, "test/loss": test_loss}, step=epoch)
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), save_to+f"/{k:.0f}k_{epoch+1}.pth")
    torch.save(model.state_dict(), save_to+f"/{k:.0f}k_{epoch+1}.pth")
    plt.title(f"{model_size} {k:.0f}k")
    plt.plot(range(epochs), train_losses, label="train")
    plt.plot(range(epochs), test_losses, label="test")
    plt.savefig(save_to+f"/{k:.0f}k_{epoch+1}.png")
    plt.legend()
    plt.show()
    
    # visualize_action(10)
    # uci = ["d7e8q", "a2b1r", "g7h8b", "h2g1q", "a2a1q"]
    # from state import encode_uci
    # for u in uci:
    #     print(f"START {u}")
    #     assert decode_action(encode_uci(u)) == u
    # # print(decode_action(encode_uci("d7qe8"))=="d7qe8")