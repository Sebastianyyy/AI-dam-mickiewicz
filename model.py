import torch
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class Config:
    len_seq:int
    vocab_size:int
    n_embedding:int
    hidden:int
    


class NGramModel(torch.nn.Module):
    def __init__(self, len_seq, vocab_size, n_embedding, hidden=200):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, n_embedding)
        self.linear1 = torch.nn.Linear(len_seq*n_embedding, hidden)
        self.tanh1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(hidden, vocab_size)
        self.soft = torch.nn.Softmax()

    def forward(self, x):
        embeds = self.embedding(x)
        flat = embeds.view(embeds.shape[0], -1)
        val = self.linear1(flat)
        tanh = self.tanh1(val)
        logits = self.linear2(tanh)
        return logits


class Bigram(torch.nn.Module):
    def __init__(self, vocab_size,chars):
        super().__init__()
        self.weight_matrix = torch.zeros((vocab_size, vocab_size))
        self.vocab_size = vocab_size
        self.occurence_ratio = torch.zeros((vocab_size, vocab_size))
        self.chars=chars
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
