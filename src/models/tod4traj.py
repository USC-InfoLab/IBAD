"""TOD4Traj's contrastive activity encoder (Zhan et al., ACM SIGSPATIAL 2024).

Port of the reference ``ContrastiveModel`` with its trajectory (lat/lon) stream
removed, so only the activity stream remains and no location reaches the model —
the same information IBAD sees. An agent's days are concatenated into one token
sequence of activities, encoded by a transformer, and each day's embedding is
the sum of its tokens' encodings. Training is contrastive over days: two days
of the same agent on the same weekday are a positive pair, days on other
weekdays of a different agent are negatives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TOD4TrajEncoder(nn.Module):
    """Activity tokens of several days -> one L2-normalised embedding per day."""

    def __init__(
        self,
        n_activities: int,
        input_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 64,
        num_layers: int = 4,
        nhead: int = 4,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(n_activities + 1, input_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.embedding.weight)
        layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim,
            dropout=0.0, batch_first=True, bias=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, T, S)`` activity ids (0 = padding) -> ``(B, T, output_dim)``."""
        batch, days, slots = tokens.shape
        encoded = self.head(self.transformer(self.embedding(tokens.reshape(batch, days * slots))))
        daily = encoded.view(batch, days, slots, self.output_dim).sum(dim=2)
        return F.normalize(daily, p=2.0, dim=-1)


def contrastive_loss(daily: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
    """The reference day-of-week contrastive objective.

    For each day, a softmax over the agent's days: those on the same weekday
    (the day itself included) are the positives, and the other-weekday slots are
    filled with days of the next agent in the batch as negatives. Padded days
    (``dow < 0``) join neither set.
    """
    valid = dow >= 0
    onehot = F.one_hot(dow.clamp(min=0).long(), num_classes=7).float() * valid.unsqueeze(-1)
    same = torch.bmm(onehot, onehot.transpose(1, 2))
    pair_valid = (valid.unsqueeze(2) & valid.unsqueeze(1)).float()
    other = (1.0 - same) * pair_valid

    within = torch.bmm(daily, daily.transpose(1, 2))
    across = torch.bmm(daily, torch.roll(daily, 1, dims=0).transpose(1, 2))
    logits = same * within + other * across

    per_day = F.cross_entropy(logits, same, reduction="none")
    day_valid = (pair_valid.sum(dim=1) > 0).float()
    return (per_day * day_valid).sum() / day_valid.sum().clamp(min=1.0)
