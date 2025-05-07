import chess
from typing import Optional
import numpy as np

import dtypes

alphabets = "abcdefgh" # row:num / col:alphabets
basics = "pnbrqk"
action_space_size = 4272
pieces = {piece:i for i, piece in enumerate(basics + basics.upper())}
itop = {i:piece for piece, i in pieces.items()}
specials = ["<me>", "<opponent>", "<board_start>", "<board_end>", "<row_end>", "<empty>", "<legal_moves>", "<white_king>", "<white_queen>", "<black_king>", "<black_queen>", "<pad>"]
special_tokens = {special:i+len(pieces.values()) for i, special in enumerate(specials)}
tokens_without_actions = {k: v+action_space_size for k, v in (pieces | special_tokens).items()}

PAD_TOK = "<pad>"

ME_TOK = "<me>"
OPPONENT_TOK = "<opponent>"
ROWEND_TOK = "<row_end>"
EMPTY_TOK = "<empty>"
BOARD_START_TOK = "<board_start>"
BOARD_END_TOK = "<board_end>"
LEGAL_MOVES = "<legal_moves>"

WHITE_KINGSIDE_CASTLING = "<white_king>"
WHITE_QUEENSIDE_CASTLING = "<white_queen>"
BLACK_KINGSIDE_CASTLING = "<black_king>"
BLACK_QUEENSIDE_CASTLING = "<black_queen>"

# convert action into UCI format movement
def decode_action(action: dtypes.Action, verbose=False) -> dtypes.UCI:
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
    if verbose:
        return from_square+to_square, (from_row, from_col, to_row, to_col)
    return from_square + to_square

def break_down_uci(move: dtypes.UCI):
    def _encode(square): # (row, col, promotion type(0~4))
        encoded = [int(square[1])-1, alphabets.index(square[0]), 0]
        if len(square) == 3:
            encoded[2] = pieces[square[2]]
        return encoded
    from_row, from_col, _ = _encode(move[:2])
    to_row, to_col, promotion = _encode(move[2:])
    return from_row, from_col, to_row, to_col, promotion

# Return index of UCI move action in action space.
def encode_uci(move: dtypes.UCI) -> dtypes.Action:
    from_row, from_col, to_row, to_col, promotion = break_down_uci(move)
    indexified = int(f'0o{from_row}{from_col}{to_row}{to_col}', 8) # 8x8 chess board
    if promotion != 0:
        return 8**4 + 44*(promotion-1) + 22*(from_row//6) + 3*from_col + (to_col-from_col)
    return indexified

decoded_actions = {decode_action(action): action for action in range(action_space_size)}
token_to_index: dict[dtypes.Token, int] = tokens_without_actions | decoded_actions
index_to_token: dict[int, dtypes.Token] = {index:token for token, index in token_to_index.items()}

vectorize = lambda token: token_to_index[token]
tokenize = lambda index: index_to_token[index]

def vectorize_fen(fen) -> dtypes.Vector:
    # 12channels + additional ones?
        # Castling rights (4 channels: kingside)
        # queenside for each color)
        # En passant target square (1 channel)
        # Turn indicator (1 channel: whose move it is)
    # rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR<opponent>e2e4<me>...
    buffer = []
    buffer.append(vectorize(BOARD_START_TOK))
    # fen = "/".join(fen.split("/")[::-1]) 
    for token in fen:
        if str.isdigit(token):
            buffer.extend([vectorize(EMPTY_TOK)] * int(token))
            continue
        if token == "/":
            buffer.append(vectorize(ROWEND_TOK))
            continue
        buffer.append(vectorize(token))
    buffer.append(vectorize(BOARD_END_TOK))
    return buffer

def tokenize_vector(vector) -> str:
    tokens = map(tokenize, vector)
    buffer = ""
    cur = 0
    for token in tokens:
        if token == EMPTY_TOK: 
            cur += 1
            continue
        elif cur != 0: 
            buffer += str(cur)
            cur = 0
        if token == ROWEND_TOK: 
            buffer += "/" 
            continue
        if token in [BOARD_START_TOK, BOARD_END_TOK]: continue
        buffer += token
    else:
        if cur != 0: buffer += str(cur)
    return buffer

def visualize_action(action: dtypes.Action, perspective=chess.WHITE):
    _, (from_row, from_col, to_row, to_col) = decode_action(action, verbose=True)
    space = [[0]*8 for _ in range(8)]
    space[from_row][from_col] = -1
    space[to_row][to_col] = 1
    if perspective == chess.WHITE:
        space = space[::-1] # white player perspective rendering
    for row in space:
        print(" ".join([f"{n:>2}" for n in row]))

class State():
    def __init__(self, board=None):
        self.board = chess.Board()
        if board is not None:
            self.board = board
        self.cur: Optional[dtypes.SAN] = None
        self._live = False
    def __repr__(self) -> str:
        return ["".join(["." * int(s) if s.isdigit() else s for s in line]) for line in self.board.board_fen().split("/")  ]
    @staticmethod
    def from_fen(fen: str):
        (board := chess.Board()).set_fen(fen)
        return State(board)
    def get_castling_rights(self) -> dtypes.Vector:
        return [
            vectorize(WHITE_KINGSIDE_CASTLING) if self.board.has_kingside_castling_rights(chess.WHITE) else 0,
            vectorize(WHITE_QUEENSIDE_CASTLING) if self.board.has_queenside_castling_rights(chess.WHITE) else 0,
            vectorize(BLACK_KINGSIDE_CASTLING) if self.board.has_kingside_castling_rights(chess.BLACK) else 0,
            vectorize(BLACK_QUEENSIDE_CASTLING) if self.board.has_queenside_castling_rights(chess.BLACK) else 0
        ]
    def serialize(self, mode="sequence") -> dtypes.Vector:
        assert mode in ["sequence", "cnn"], "Please choose the appropriate mode."
        if mode == "sequence":
            vector_legal_moves = [vectorize(LEGAL_MOVES)] + self.get_legel_actions()
            vector_fen = vectorize_fen(self.board.board_fen())
            vector_castling_rights = [v for v in self.get_castling_rights() if v != 0]
            return vector_fen + vector_legal_moves + vector_castling_rights
        elif mode == "cnn":
            bstate = np.zeros(64, np.uint16)
            for i in range(64):
                pp = self.board.piece_at(i)
                if pp is not None:
                    bstate[i] = 2 ** pieces[pp.symbol()] # 0~11
            state = np.zeros((15, 64))
            for i in range(12):
                state[i] = (bstate>>i)&1
            if self.board.has_kingside_castling_rights(chess.WHITE):
                assert bstate[7]==2**pieces["R"]
                state[12][7] = 1
            if self.board.has_queenside_castling_rights(chess.WHITE):
                assert bstate[0]==2**pieces["R"]
                state[12][0] = 1
            if self.board.has_kingside_castling_rights(chess.BLACK):
                assert bstate[63]==2**pieces["r"]
                state[13][63] = 1
            if self.board.has_queenside_castling_rights(chess.BLACK):
                assert bstate[56]==2**pieces["r"]
                state[13][56] = 1
            state[14] = self.board.turn * 1.0
            state = state.reshape(15, 8, 8)
            return state
    @staticmethod
    def deserialize(vector: dtypes.Vector, mode="sequence") -> str:
        assert mode in ["sequence", "cnn"], "Please choose the appropriate mode." 
        if mode == "sequence":
            return tokenize_vector(vector)
        elif mode == "cnn":
            return NotImplementedError
    def push(self, move: chess.Move):
        if self._live:
            self.cur = self.board.san(move)
        self.board.push(move)
    def push_uci(self, uci: dtypes.UCI):
        move = chess.Move.from_uci(uci)
        self.push(move)
    def undo(self):
        self.board.pop()
    @property
    def current_move(self) -> dtypes.SAN:
        return self.cur
    def get_legel_actions(self) -> dtypes.Actions:
        legal_moves = [
            encode_uci(move.uci())
            for move in self.board.legal_moves
        ]
        return legal_moves
    def reset(self):
        self.board.reset()
        self.cur = None
        self._live = False
    def copy(self): return self.board.copy()
    def clone(self): return State(self.copy())
    def __str__(self):
        return str(self.board)
    def game_over(self) -> bool:
        return self.board.is_game_over()
    def game_result(self) -> str:
        return self.board.outcome().result()
    def detach(self): self._live = False
    def attach(self): self._live = True

if __name__ == "__main__":
    # print(token_to_index)
    print(len(token_to_index))
    state = State()
    print(state.serialize())
    print(State.deserialize(state.serialize()) == state.board.board_fen())
    print(state.serialize("cnn"))
    print(state)