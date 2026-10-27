"""ICAD — a self-supervised autoregressive model of an agent's visit sequence.

Port of ICAD ("Self-Supervised Autoregressive Approach for Multi-Context Anomaly
Detection in Human Mobility Data"), extended with a per-agent identity
embedding. The reference model embeds and predicts each visit's H3 region; here
the region is replaced by the visit's activity, so ICAD sees agent, activity and
time — the same information IBAD sees, and no location.

An agent's visits form one causal sequence, with a separator token closing each
day. Each visit is embedded as

    [agent embedding | activity embedding | Time2Vec(arrival) | Time2Vec(departure)]

a causal transformer encodes the history, and three heads predict the next
visit in turn, each conditioned on more of it (teacher forcing): its activity
(from the agent), its travel time since the previous departure (from agent and
activity), and its departure time of day (from agent, activity and arrival).
"""

import math

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

PAD, SEPARATOR = 0, 1      # activity ids reserved for padding and day separators
N_SPECIAL = 2


class Time2Vec(nn.Module):
    """Learned periodic time encoding (Kazemi et al., 2019)."""

    def __init__(self, d_embed: int):
        super().__init__()
        self.linear = nn.Linear(1, d_embed)

    def forward(self, x: Tensor) -> Tensor:
        h = self.linear(x.unsqueeze(-1))
        return torch.cat([h[..., :1], torch.sin(h[..., 1:])], dim=-1)


class PositionalEncoding(nn.Module):
    """Sinusoidal position encoding with dropout."""

    pe: Tensor

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class CausalMemoryDecoder(nn.Module):
    """Cross-attention to the causal history, then a feed-forward block (post-norm).

    Position *i* of the target may attend memory positions ``<= i`` only.
    """

    def __init__(self, d_model: int, num_heads: int, d_feedforward: int, d_memory: int,
                 dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                               batch_first=True, kdim=d_memory, vdim=d_memory)
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.linear2 = nn.Linear(d_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        steps = memory.size(1)
        mask = torch.triu(torch.ones(steps, steps, dtype=torch.bool, device=memory.device), 1)
        attended, _ = self.attention(tgt, memory, memory, attn_mask=mask, need_weights=False)
        x = self.norm1(tgt + self.dropout1(attended))
        feed = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.norm2(x + self.dropout2(feed))


class GMMHead(nn.Module):
    """Parameters of a 1-D Gaussian mixture."""

    def __init__(self, d_model: int, num_gaussians: int):
        super().__init__()
        self.weight = nn.Sequential(nn.Linear(d_model, num_gaussians), nn.Softplus())
        self.loc = nn.Linear(d_model, num_gaussians)
        self.scale = nn.Sequential(nn.Linear(d_model, num_gaussians), nn.Softplus())

    def forward(self, x: Tensor, eps: float = 1e-6) -> D.MixtureSameFamily:
        weight = self.weight(x) + eps
        scale = torch.clamp(self.scale(x) + eps, max=100).clamp(min=1e-6)
        return D.MixtureSameFamily(
            D.Categorical(weight / weight.sum(dim=-1, keepdim=True)),
            D.Normal(self.loc(x), scale),
        )


class ICAD(nn.Module):
    """Predicts each visit's activity, travel time and departure time from its history."""

    def __init__(
        self,
        n_agents: int,
        n_activities: int,
        num_heads: int = 2,
        num_layers: int = 4,
        num_gaussians: int = 3,
        d_feedforward: int = 32,
        d_embed: int = 32,
    ):
        super().__init__()
        self.d_embed = d_embed
        d_model = 4 * d_embed
        self.agent_embedding = nn.Embedding(n_agents, d_embed)
        self.activity_embedding = nn.Embedding(n_activities + N_SPECIAL, d_embed)
        self.time2vec = Time2Vec(d_embed)

        self.position = PositionalEncoding(d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True), num_layers,
        )
        self.activity_decoder = CausalMemoryDecoder(d_embed, num_heads, d_feedforward, d_model)
        self.activity_head = nn.Linear(d_embed, n_activities + N_SPECIAL)
        self.travel_decoder = CausalMemoryDecoder(2 * d_embed, num_heads, d_feedforward, d_model)
        self.travel_head = GMMHead(2 * d_embed, num_gaussians)
        self.departure_decoder = CausalMemoryDecoder(3 * d_embed, num_heads, d_feedforward, d_model)
        self.departure_head = GMMHead(3 * d_embed, num_gaussians)

    def forward(self, agent: Tensor, activity: Tensor, arrival: Tensor,
                departure: Tensor) -> dict:
        """``(B, L)`` inputs -> predictions for positions ``1..L-1``.

        Sequences are right-padded, so under the causal masks no real visit ever
        attends to padding and an agent's scores do not depend on its batch.
        """
        embedded = torch.cat([
            self.agent_embedding(agent),
            self.activity_embedding(activity),
            self.time2vec(arrival),
            self.time2vec(departure),
        ], dim=-1)
        steps = embedded.size(1) - 1
        causal = torch.triu(torch.ones(steps, steps, dtype=torch.bool, device=embedded.device), 1)
        memory = self.encoder(self.position(embedded[:, :-1]), mask=causal, is_causal=True)
        target = embedded[:, 1:]
        d = self.d_embed
        return {
            "activity": self.activity_head(self.activity_decoder(target[..., :d], memory)),
            "travel": self.travel_head(self.travel_decoder(target[..., :2 * d], memory)),
            "departure": self.departure_head(self.departure_decoder(target[..., :3 * d], memory)),
        }


def icad_loss(output: dict, activity: Tensor, travel: Tensor, departure_tod: Tensor,
              real: Tensor) -> Tensor:
    """Activity cross-entropy plus the two GMM negative log-likelihoods over real visits."""
    def gmm_nll(mixture: D.MixtureSameFamily, values: Tensor) -> Tensor:
        return -mixture.log_prob(values)[real].mean()

    return (F.cross_entropy(output["activity"][real], activity[real])
            + gmm_nll(output["travel"], travel)
            + gmm_nll(output["departure"], departure_tod))


_TOPK = 3


def _mode_margin(mixture: D.MixtureSameFamily, values: Tensor) -> Tensor:
    """How far below the mixture's best mode the observed value's log-density falls."""
    components = mixture.component_distribution
    log_at_modes = components.log_prob(components.mode)
    log_pi = mixture.mixture_distribution.logits.log_softmax(dim=-1)
    return (log_at_modes + log_pi).logsumexp(dim=-1) - mixture.log_prob(values)


def visit_scores(output: dict, activity: Tensor, travel: Tensor,
                 departure_tod: Tensor, weights: tuple[float, float, float]) -> Tensor:
    """The reference ``relative`` per-visit anomaly score.

    Activity: mean log-probability of the model's top-3 activities minus that of
    the observed one. Times: the mode margin of each GMM. ``weights`` combine the
    activity, travel-time and departure-time terms. Higher = more anomalous.
    """
    logp = output["activity"].clone()
    logp[..., :N_SPECIAL] = float("-inf")
    logp = logp.log_softmax(dim=-1)
    observed = logp.gather(-1, activity.unsqueeze(-1)).squeeze(-1)
    activity_score = logp.topk(min(_TOPK, logp.size(-1)), dim=-1).values.mean(dim=-1) - observed
    return (weights[0] * activity_score
            + weights[1] * _mode_margin(output["travel"], travel)
            + weights[2] * _mode_margin(output["departure"], departure_tod))
