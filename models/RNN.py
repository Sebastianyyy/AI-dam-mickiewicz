import torch
import Config

class RNN(torch.nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(config.vocab_size, config.embedding_dim)
        self.RNNCell = RNNCell(config=config)

    def forward(self, X):
        X = self.emb(X)
        logits = RNNCell(X)


class RNNCell(torch.nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.hidden_size = config.hidden
        self.batch_size = config.batch_size
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
