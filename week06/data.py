import re
import chess
import chess.pgn
import io
import tqdm
import numpy as np

username = "yoonhero"
path = "./my_chess.pgn"
basics = "pnbrqk"
pieces = {piece:i for i, piece in enumerate(basics + basics.upper())}
itop = {i:piece for piece, i in pieces.items()}
specials = ["<me>", "<opponent>", "<board_start>", "<board_end>", "<row_end>", "<empty>"]
special_tokens = {special:i+len(pieces.values()) for i, special in enumerate(specials)}
tokens = pieces | special_tokens

ME_TOK = "<me>"
OPPONENT_TOK = "<opponent>"
ROWEND_TOK = "<row_end>"
EMPTY_TOK = "<empty>"
BOARD_START_TOK = "<board_start>"
BOARD_END_TOK = "<board_end>"

def tokenize_fen(fen):
    # 12channels + additional ones?
        # Castling rights (4 channels: kingside)
        # queenside for each color)
        # En passant target square (1 channel)
        # Turn indicator (1 channel: whose move it is)
    # too many! -> move on to token tranformer!
    # rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
    buffer = []
    buffer.append(tokens[BOARD_START_TOK])
    for char in fen:
        if str.isdigit(char):
            buffer.extend([tokens[EMPTY_TOK]] * int(char))
            continue
        if char == "/":
            buffer.append(tokens[ROWEND_TOK])
            continue
        buffer.append(tokens[char])
    buffer.append(tokens[BOARD_END_TOK])
    return buffer

alphabets = "abcdefgh" # row:num / col:alphabets
# Return index of UCI move action in action space.
def encode_uci(move):
    def _encode(square): # (row, col, promotion type(0~4))
        encoded = [int(square[1])-1, alphabets.index(square[0]), 0]
        if len(square) == 3:
            encoded[2] = pieces[square[2]]
        return encoded
   
    try:
        from_row, from_col, _ = _encode(move[:2])
        to_row, to_col, promotion = _encode(move[2:])
    except ValueError:
        return None
    indexified = int(f'0o{from_row}{from_col}{to_row}{to_col}', 8) # 8x8 chess board
    if promotion != 0:
        return 8**4 + 44*(promotion-1) + 22*(from_row//6) + 3*from_col + (to_col-from_col)
    return indexified

# convert action into UCI format movement
def decode_action(action):
    promotion = ""
    from_row, from_col, to_row, to_col = None, None, None, None
    if action >= 8**4:
        action %= 8**4
        promotion_index = action // 44 + 1
        promotion = itop[promotion_index]
        action %= 44
        from_row = 6 if bool(action // 22) else 1
        to_row = 7 if from_row == 6 else 0
        action %= 22
        for i in range(8):
            if action+1 > sum([2, 3, 3, 3, 3, 3, 3, 2][:i+1]):
                continue
            from_col = i
            to_col = action - 2*i
            break
    else:
        from_row, from_col, to_row, to_col = map(int, f"{action:04o}") # same as oct(function)
    from_square = alphabets[from_col] + str(from_row+1)
    to_square = alphabets[to_col] + str(to_row+1) + promotion
    return from_square + to_square

def parse_game(pgn):
    # Protocols                         Type               Unambiguity                   Human friendly
    # UCI(universal chess interface)    Only movement(DAG) None                          No(Engine)
    # SAN(standard algebraic notation)  Including Actions  Probably(Additional notation) Yes(PGN)
        # types: x(capture), +(check), #(checkmate)
        # pawn capture -> exd5 / promotion -> e8=Q
    # game -> 1. e3 {[%clk 0:09:58.9]} 1... e5 {[%clk 0:09:55.2]} ... 1-0

    #board.push_san("e4") # san(standard algebraic notation) --> only Result vs uci(universal chess interface) --> DAG
    # moves = [move.split(" ")[0] for move in re.split(r'\d+\.+ ', game)[1:]]
    # for move in moves:
    #     board.push_san(move)
    #     str(board.move_stack[-1])
    #     fens.append(board.board_fen())
    stream = io.StringIO(pgn)
    game = chess.pgn.read_game(stream)
    board = game.board()
    fens = []
    moves = []
    for move in game.mainline_moves():
        fens.append(tokenize_fen(board.board_fen()))
        board.push(move)
        encoded_move = encode_uci(str(move))
        if encoded_move is None:
            return [], []
        # assert str(move) == decode_action(encode_uci(str(move)))
        moves.append(encoded_move)

    return fens, moves

def doyouwin(line):
    am_i_white = line[4].split('"')[-2] == username
    score = line[6].split('"')[-2]
    if score == "1/2-1/2":
        return 0.5
    else:
        is_white_win = int(score == "1-0")
        return is_white_win if am_i_white else 1-is_white_win

def load_data(path):
    games = []
    with open(path, "r", encoding="utf-8") as f:
        # games = [(parse_game((line:=lines.split("\n"))[-1]), doyouwin(line)) for lines in f.read().split("\n\n\n")[:-1]]
        for line in tqdm.tqdm(f.read().rstrip().split("\n\n\n")):
            games.append(parse_game(line))
        
    return games


if __name__ == "__main__":
    data = load_data(path)
    uci = ["d7e8q", "a2b1r", "g7h8b", "h2g1q", ]
    for u in uci:
        print(f"START {u}")
        assert decode_action(encode_uci(u)) == u
    # print(decode_action(encode_uci("d7qe8"))=="d7qe8")