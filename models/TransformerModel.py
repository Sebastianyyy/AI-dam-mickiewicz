import torch
import math
from einops import rearrange

class TransformerModel(torch.nn.Module):

    def __init__(self, config):
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
    @torch.no_grad()
    def sample(self,text,l,temperature):
        return NotImplemented


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
