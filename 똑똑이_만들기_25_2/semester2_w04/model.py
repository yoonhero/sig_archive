import torch
import torch.nn as nn
import torch.nn.functional as F

device = "mps"

class TinyRNNLayer(nn.Module):
    def __init__(self, emb_size, hidden_size, total_tokens):
        super().__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.total_tokens = total_tokens
        
        self.model = nn.Sequential(
            nn.Linear(emb_size+hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, emb_size+hidden_size)           
        )

    def forward(self, x, hidden=None):
        B, T, C = x.shape
        if hidden == None:
            hidden = torch.zeros((B, self.hidden_size), device=device)
        outs = []
        for t in range(T):
            agg = torch.cat([x[:,t,:], hidden], dim=-1)
            out, hidden = torch.split(self.model(agg), [self.emb_size, self.hidden_size], dim=-1)
            hidden = torch.tanh(hidden)
            outs.append(out)
        outs = torch.stack(outs, dim=1)
        return outs, hidden

class TinyRNN(nn.Module):
    def __init__(self, layers, emb_size, hidden_size, total_tokens):
        super().__init__()
        self.n_layers = layers
        self.layers = nn.ModuleList(
            [TinyRNNLayer(emb_size, hidden_size, total_tokens) for _ in range(layers)]
        )
        # self.layernorm = nn.LayerNorm()
        self.emb_table = nn.Embedding(total_tokens, emb_size)
        self.act = nn.Tanh()
        self.proj = nn.Linear(emb_size, total_tokens)

    def forward(self, x):
        x = self.emb_table(x)
        for layer in self.layers:
            out, hi = layer(x)
            x = x + out
        return self.proj(self.act(x))