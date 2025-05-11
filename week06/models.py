import torch.nn
import math

from state import total_tokens, vectorize, PAD_TOK

board_length = 73 # only evaluates the board.

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
        attn = q @ k.transpose(-2, -1) * (1 / math.sqrt(C))
        attn = attn.masked_fill(~self.mask[:,:T,:T], float("-inf"))
        attn = torch.softmax(attn, dim=-1)
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
    def __init__(self, embedding, n_layer):
        super().__init__()
        self.total_tokens = total_tokens + 2
        self.emb = torch.nn.Embedding(self.total_tokens, embedding)
        self.blocks = torch.nn.ModuleList([AttentionBlock(embedding, i) for i in range(n_layer)])
        self.norm = torch.nn.LayerNorm(embedding)
        self.linear = torch.nn.Linear(embedding, self.total_tokens)
        self.act = torch.nn.ReLU()
    def forward(self, x, train=False, attention_masks=None):
        x = self.emb((start:=x))
        for block in self.blocks:
            x = block(x)
        x = self.act(self.norm(x))
        logits = self.linear(x)
        if train:
            if attention_masks is None:
                return torch.nn.functional.cross_entropy(logits[:, board_length:-1].contiguous().view(-1, self.total_tokens), start[:, board_length+1:].contiguous().view(-1), ignore_index=vectorize(PAD_TOK), reduction="mean")
            else:
                B = attention_masks.shape[0]
                attention_masks = attention_masks.view(-1)
                logits = logits[range(B), attention_masks-1].contiguous().view(-1, self.total_tokens)
                start = start[range(B), attention_masks].contiguous().view(-1)
                return torch.nn.functional.cross_entropy(logits, start, ignore_index=vectorize(PAD_TOK), reduction="mean")
        return logits

if __name__ == "__main__":
    from data import load_games, make_dataloader
    from data import models
    model = AttentionPolicy(*models["small"])
    # w = torch.load("./model/medium/100k_20.pth", weights_only=True)
    w = torch.load("./model/small/100k_20.pth", weights_only=True)
    w = {k: v for k, v in w.items() if "mask" not in k}
    model.load_state_dict(w, strict=False) # terrible value network!
    username = "yoonhero"
    path = "./data/DATABASE4U.pgn"
    mode = "sequence" # or cnn
    save_path = f"./data/processed/database4u_withturn_{mode}.npz"
    vectors, actions, attention_masks, results = load_games(path, mode, max_length=160, save_path=save_path)
    dataloader, test_dataloader = make_dataloader(vectors[:10000], results[:10000], attention_masks[:10000])
    x, _, mask = next(iter(dataloader))
    print(mask.shape)
    print(model(x, True, mask))
