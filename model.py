import torch
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import math
from einops import rearrange


@dataclass
class Config:
    seq_len: int
    vocab_size:int
    embedding_dim:int
    hidden:int
    dropout:float
    n_layers:int 
    n_head:int
    batch_size:int


class NGramModel(torch.nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.embedding_dim)
        self.linear1 = torch.nn.Linear(
            config.seq_len*config.embedding_dim, config.hidden)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(config.hidden, config.vocab_size)
        self.soft = torch.nn.Softmax()

    def forward(self, x):
        embeds = self.embedding(x)
        flat = embeds.view(embeds.shape[0], -1)
        val = self.linear1(flat)
        tanh = self.tanh1(val)
        logits = self.linear2(tanh)
        return logits


class Bigram(torch.nn.Module):
    def __init__(self, config:Config,chars):
        super().__init__()
        self.weight_matrix = torch.zeros(
            (config.vocab_size, config.vocab_size))
        self.vocab_size = config.vocab_size
        self.occurence_ratio = torch.zeros(
            (config.vocab_size, config.vocab_size))
        self.chars=config.chars
    def __call__(self, X):
        for i, j in zip(X[:-1], X[1:]):
            self.weight_matrix[i][j] += 1
        sumation = self.weight_matrix.sum(dim=1, keepdim=True)
        self.occurence_ratio = (self.weight_matrix+1) / \
            (sumation+self.vocab_size)

    def plot(self):
        fig1, ax = plt.subplots(figsize=(40, 40))
        ax.imshow(self.weight_matrix,  cmap='hot', interpolation='nearest')
        plt.xticks(np.arange(0, self.vocab_size), self.chars)
        plt.yticks(np.arange(0, self.length), self.chars)
        plt.xticks(fontsize=45)
        plt.yticks(fontsize=45)
        for (j, i), label in np.ndenumerate(self.weight_matrix):
            ax.text(i, j, label, ha='center', va='center')
        plt.show()

    """ def sample(self, string, length_sequence):
        letter = string[-1] if len(string) != 0 else " "
        for i in range(length_sequence):
            x = torch.tensor(encode(letter), dtype=torch.long)
            probs = self.occurence_ratio[x.item(), :]
            multinomial = torch.multinomial(probs, 1, replacement=True).item()
            letter = decode([multinomial])
            string += letter
        return string"""

    def loss(self, X, y):
        length_sequence = X.size(dim=0)
        probs = torch.sum(torch.log(self.occurence_ratio[X, y]))
        return -probs/length_sequence


class RNN(torch.nn.Module):
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(config.vocab_size, config.embedding_dim)
        self.RNNCell=RNNCell(config=config)
    def forward(self,X):
        X=self.emb(X)
        logits=RNNCell(X)
        

class RNNCell(torch.nn.Module):
    def __init__(self,config:Config) -> None:
        super().__init__()
        self.hidden_size=config.hidden
        self.batch_size=config.batch_size
        self.wxh = torch.nn.Linear(config.embedding_dim, config.hidden)
        self.whh = torch.nn.Linear(config.hidden, config.hidden)
        self.wyh = torch.nn.Linear(config.hidden, config.vocab_size)

    def forward(self, X):
        self.h = torch.zeros(
            self.batch_size, self.hidden_size, requires_grad=False)

        output = []
        for time in range(X.shape[1]):
            X_time = X[:, time, :]
            xh = self.wxh(X_time)
            hh = self.whh(self.h)
            self.h = xh+hh
            out = self.wyh(self.h)
            output.append(out)
        return torch.stack(output, dim=1)

class TransformerModel(torch.nn.Module):

    def __init__(self,config:Config):
        super().__init__()
        self.emb = torch.nn.Embedding(config.vocab_size, config.embedding_dim)
        self.encoders = torch.nn.ModuleList(
            [Encoder(config.n_head, config.embedding_dim, config.dropout, config.seq_len) for _ in range(config.n_layers)])
        self.out = torch.nn.Linear(
            config.embedding_dim, config.vocab_size, bias=False)
        self.pos = self.positional_encoder(
            config.embedding_dim, config.seq_len)

    def positional_encoder(self, embedding_dim, seq_len):
        div_term = torch.exp(torch.arange(0, embedding_dim, 2)
                             * -(math.log(1000) / embedding_dim))
        k = torch.arange(0, seq_len).view(-1, 1)
        pos = torch.zeros(seq_len, embedding_dim)
        pos[:, 0::2] = torch.sin(k*div_term)
        pos[:, 1::2] = torch.cos(k*div_term)
        pos = pos.unsqueeze(0)
        return pos

    def forward(self, x):
        x = self.emb(x)+self.pos
        for enc in self.encoders:
            x = enc(x)
        logits = self.out(x)
        return logits


class Encoder(torch.nn.Module):
    def __init__(self, n_head, embedding_dim, dropout, seq_len):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(embedding_dim)
        self.attention = SelfAttention(n_head, embedding_dim, dropout, seq_len)
        self.ln_2 = torch.nn.LayerNorm(embedding_dim)
        self.mlp = torch.nn.ModuleList([torch.nn.Linear(embedding_dim, embedding_dim*4),
                                        torch.nn.Linear(
            embedding_dim*4, embedding_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout)
        ])

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        r = self.ln_2(x)
        for layer in self.mlp:
            r = layer(r)
        x = r+x
        return x


class SelfAttention(torch.nn.Module):
    def __init__(self, n_head, embedding_dim, dropout, seq_len):
        super().__init__()
        self.c_attn = torch.nn.Linear(embedding_dim, 3*embedding_dim)
        self.c_proj = torch.nn.Linear(embedding_dim, embedding_dim)
        self.attn_dropout = torch.nn.Dropout(dropout)

        self.n_head = n_head
        self.embedding_dim = embedding_dim
        self.register_buffer("bias", torch.tril(torch.ones(seq_len, seq_len))
                             .view(1, 1, seq_len, seq_len))

    def forward(self, x):
        B, L, N = x.shape
        q, k, v = self.c_attn(x).split(self.embedding_dim, dim=-1)

        q = rearrange(q, 'b l (h1 h2) -> b h1 l h2',
                      h2=N//self.n_head, h1=self.n_head)
        k = rearrange(k, 'b l (h1 h2) -> b h1 l h2',
                      h2=N//self.n_head, h1=self.n_head)
        v = rearrange(v, 'b l (h1 h2) -> b h1 l h2',
                      h2=N//self.n_head, h1=self.n_head)

        attention = (q@k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
        masked = attention.masked_fill_(
            self.bias[:, :, :L, :L] == 0, float('-inf'))

        softmax = torch.nn.functional.softmax(masked, dim=-1)

        y = softmax@v
        y = y.transpose(1, 2).contiguous().view(B, L, N)
        y = self.attn_dropout(self.c_proj(y))

        return y
