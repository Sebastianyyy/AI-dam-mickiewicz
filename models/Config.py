from dataclasses import dataclass
import torch
@dataclass
class Config:
    seq_len: int
    vocab_size: int
    embedding_dim: int
    hidden: int
    dropout: float
    n_layers: int
    n_head: int
    batch_size: int
    device: torch.device