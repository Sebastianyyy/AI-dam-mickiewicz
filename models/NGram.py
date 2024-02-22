import torch
import Config

class NGramModel(torch.nn.Module):
    def __init__(self, config: Config):
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
