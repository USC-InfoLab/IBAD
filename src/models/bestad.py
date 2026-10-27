"""SJCAE — the clustering autoencoder at the core of the BeSTAD baseline.

Port of the Sequence-to-sequence Joint Clustering AutoEncoder from the BeSTAD
reference implementation (Behavior-aware Spatio-Temporal Anomaly Detection).
The reference fuses a temporal feature branch with a multi-scale H3-OSM spatial
branch; here the spatial branch is fed the stay's activity instead, so BeSTAD
sees time of day and activity — the same information IBAD sees, and no
location. The architecture is otherwise unchanged: two BatchNorm projection
encoders feed a fusion layer, a bidirectional LSTM encodes the day's stays into
a latent ``z_mean`` and a Gumbel-softmax cluster assignment, and an LSTM decoder
reconstructs both feature blocks from the cluster-centred latent.

A day's argmax cluster is its behavioural template; the detector built on top
(``src/detectors/bestad.py``) scores a day by how its template departs from the
agent's normal template profile.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _BNProjection(nn.Module):
    """Per-step Linear -> BatchNorm -> ReLU -> Dropout."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = x.size()
        return self.projection(x.reshape(-1, x.size(-1))).view(batch, steps, -1)


def _head(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class SJCAE(nn.Module):
    """Seq2seq joint-clustering autoencoder over one day's stay sequence."""

    def __init__(
        self,
        temporal_dim: int,
        activity_dim: int,
        n_clusters: int = 6,
        lstm_dim: int = 48,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.temporal_dim = temporal_dim
        self.hidden_dim = hidden_dim
        self.lstm_dim = lstm_dim

        self.temporal_encoder = _BNProjection(temporal_dim, hidden_dim, dropout)
        self.activity_encoder = _BNProjection(activity_dim, hidden_dim, dropout)
        self.fusion = _BNProjection(2 * hidden_dim, hidden_dim, dropout)
        self.encoder_lstm = nn.LSTM(hidden_dim, lstm_dim, num_layers=2, batch_first=True,
                                    bidirectional=True, dropout=dropout)

        self.fc_z_mean = _head(2 * lstm_dim, hidden_dim, hidden_dim, dropout)
        self.fc_y = _head(2 * lstm_dim, hidden_dim, n_clusters, dropout)
        self.fc_cluster_mean = nn.Linear(n_clusters, hidden_dim)
        nn.init.xavier_uniform_(self.fc_cluster_mean.weight)
        nn.init.zeros_(self.fc_cluster_mean.bias)

        self.decoder_lstm = nn.LSTM(hidden_dim, lstm_dim, num_layers=2, batch_first=True,
                                    dropout=dropout)
        self.temporal_decoder = _head(lstm_dim, hidden_dim, temporal_dim, dropout)
        self.activity_decoder = _head(lstm_dim, hidden_dim, activity_dim, dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal_encoder(x[:, :, :self.temporal_dim])
        activity = self.activity_encoder(x[:, :, self.temporal_dim:])
        fused = self.fusion(torch.cat([temporal, activity], dim=-1))
        out, _ = self.encoder_lstm(fused)
        return out[:, -1, :]

    def decode(self, z: torch.Tensor, steps: int) -> torch.Tensor:
        batch = z.size(0)
        out, _ = self.decoder_lstm(z.unsqueeze(1).expand(batch, steps, self.hidden_dim))
        flat = out.reshape(-1, self.lstm_dim)
        return torch.cat([
            self.temporal_decoder(flat).view(batch, steps, -1),
            self.activity_decoder(flat).view(batch, steps, -1),
        ], dim=-1)

    def forward(self, x: torch.Tensor) -> dict:
        h = self.encode(x)
        z_mean = self.fc_z_mean(h)
        z_diff = z_mean + torch.randn_like(z_mean) * 0.05 if self.training else z_mean
        y_logits = self.fc_y(h)
        y = gumbel_softmax(y_logits, temperature=0.8, hard=True)
        z_center = self.fc_cluster_mean(y)
        steps = x.size(1)
        return {
            "z_mean": z_mean,
            "y_soft": F.softmax(y_logits, dim=-1),
            "x_recon": self.decode(z_center + z_diff, steps),
            "x_center": self.decode(z_center, steps),
        }

    @torch.no_grad()
    def assign(self, x: torch.Tensor) -> torch.Tensor:
        """Soft cluster assignment, without running the decoder."""
        return F.softmax(self.fc_y(self.encode(x)), dim=-1)


def gumbel_softmax(logits: torch.Tensor, temperature: float, hard: bool) -> torch.Tensor:
    """Straight-through Gumbel-softmax sample over the clusters."""
    eps = 1e-20
    noise = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
    y = F.softmax((logits + noise) / temperature, dim=-1)
    if not hard:
        return y
    y_hard = torch.zeros_like(y).scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
    return (y_hard - y).detach() + y


def sjcae_loss(
    x: torch.Tensor,
    outputs: dict,
    temporal_dim: int,
    reg_weight: float = 0.01,
    center_weight: float = 1.0,
    nent_weight: float = 0.1,
    diversity_weight: float = 0.3,
    consistency_weight: float = 0.1,
    temporal_weight: float = 1.5,
    activity_weight: float = 1.2,
    nent_max: float = 0.5,
) -> torch.Tensor:
    """The reference SJCAE objective.

    Masked reconstruction from the full and the cluster-centre latent, an L2
    penalty on the latent, and three terms on the soft assignment: a hinge on
    per-day entropy, a pull of the average assignment towards uniform (cluster
    diversity), and a reward for confident assignments. Padded stays (all-zero
    rows) are masked out of the reconstruction terms.
    """
    mask = (x.abs().sum(dim=-1) > 1e-6).float()
    n_stays = mask.sum(dim=-1).clamp(min=1).sum()

    def reconstruction(recon: torch.Tensor) -> torch.Tensor:
        temporal = F.mse_loss(recon[:, :, :temporal_dim], x[:, :, :temporal_dim], reduction="none")
        activity = F.mse_loss(recon[:, :, temporal_dim:], x[:, :, temporal_dim:], reduction="none")
        return (temporal_weight * ((temporal.sum(-1) * mask).sum() / n_stays)
                + activity_weight * ((activity.sum(-1) * mask).sum() / n_stays))

    y_soft = outputs["y_soft"]
    eps = 1e-10
    neg_entropy = (y_soft * torch.log(y_soft + eps)).sum(dim=-1)
    cluster_probs = y_soft.mean(dim=0)
    uniform = torch.ones_like(cluster_probs) / len(cluster_probs)

    return (
        reconstruction(outputs["x_recon"])
        + reg_weight * (outputs["z_mean"] ** 2).sum(dim=-1).mean()
        + center_weight * reconstruction(outputs["x_center"])
        + nent_weight * F.relu(neg_entropy + nent_max).mean()
        + diversity_weight * F.kl_div(torch.log(cluster_probs + eps), uniform, reduction="sum")
        - consistency_weight * y_soft.max(dim=-1)[0].mean()
    )
