"""IBAD — the hierarchical behavioural forecaster at the core of module 2.

Forecasts what an agent's *next* day should look like, expressed as a mixture
over the behavioural templates from module 1. A day is anomalous when what
actually happened diverges from that forecast.

Three levels:

    visits of one day  ──►  VisitEmbedder + IntraDayEncoder  ──►  day vector
    days of one agent  ──►  causal continuous-time transformer ──►  next-day logits
    logits vs observed theta ──────────────────────────────►  anomaly score

The day-level attention is causal (a day may only attend to earlier days) and
continuous-time: queries carry the *target* day's timestamp and keys the history
day's, so the forecast is conditioned on when the predicted day actually falls.
Irregular sampling therefore needs no calendar padding.

Continuous time alone never states that two Saturdays are the same kind of day,
so the target day's day-of-week is added to the query as well. The forecast is
then "what does this agent do on a Saturday" rather than "what does this agent
usually do".
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Time2Vec(nn.Module):
    """Learned periodic time encoding (Kazemi et al., 2019).

    The first channel stays linear so absolute time survives; the rest are
    sinusoids with learned frequency and phase.
    """

    def __init__(self, d_embed: int):
        super().__init__()
        self.linear = nn.Linear(1, d_embed)

    def forward(self, x: Tensor) -> Tensor:
        h = self.linear(x.unsqueeze(-1))
        return torch.cat([h[..., :1], torch.sin(h[..., 1:])], dim=-1)


class VisitEmbedder(nn.Module):
    """One visit -> one vector, from its arrival and departure time and its activity.

    Times arrive as Unix seconds (see ``training.dataset.encode_times``).
    """

    def __init__(self, n_bundles: int, d_embed: int, d_visit: int):
        super().__init__()
        self.t2v_arrival = Time2Vec(d_embed)
        self.t2v_departure = Time2Vec(d_embed)
        self.bundle_embed = nn.Embedding(n_bundles + 1, d_embed, padding_idx=0)
        self.proj = nn.Linear(3 * d_embed, d_visit, bias=False)

    def forward(self, arrival: Tensor, departure: Tensor, bundle_id: Tensor) -> Tensor:
        return self.proj(torch.cat([
            self.t2v_arrival(arrival),
            self.t2v_departure(departure),
            self.bundle_embed(bundle_id),
        ], dim=-1))


class BidirectionalAttention(nn.Module):
    """Unmasked self-attention over the visits within a day."""

    # PyTorch's memory-efficient attention kernel cannot seed a batch larger
    # than this, and flattening (B, T, V, D) -> (B*T, V, D) reaches it easily on
    # long Geolife histories, so oversized batches are attended in slices.
    _SDPA_BATCH_CAP = 65535

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: Tensor, key_padding_mask: Tensor) -> Tensor:
        n, v, d = x.shape
        q, k, val = (
            t.view(n, v, self.n_heads, self.head_dim).transpose(1, 2)
            for t in self.qkv(x).split(d, dim=-1)
        )
        attn_mask = torch.zeros(n, 1, 1, v, device=x.device, dtype=x.dtype)
        attn_mask = attn_mask.masked_fill(
            ~key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
        )

        dropout_p = self.dropout if self.training else 0.0
        pieces = []
        for start in range(0, n, self._SDPA_BATCH_CAP):
            stop = min(start + self._SDPA_BATCH_CAP, n)
            pieces.append(F.scaled_dot_product_attention(
                q[start:stop], k[start:stop], val[start:stop],
                attn_mask=attn_mask[start:stop], dropout_p=dropout_p,
            ))
        out = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]
        return out.transpose(1, 2).contiguous().view(n, v, d)


class IntraDayEncoder(nn.Module):
    """Pre-norm transformer over one day's visits."""

    def __init__(self, d_visit: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn_norms = nn.ModuleList(nn.LayerNorm(d_visit) for _ in range(n_layers))
        self.attns = nn.ModuleList(
            BidirectionalAttention(d_visit, n_heads, dropout) for _ in range(n_layers)
        )
        self.mlp_norms = nn.ModuleList(nn.LayerNorm(d_visit) for _ in range(n_layers))
        self.mlps = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d_visit, 4 * d_visit, bias=False),
                nn.GELU(approximate="tanh"),
                nn.Linear(4 * d_visit, d_visit, bias=False),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        )
        self.ln = nn.LayerNorm(d_visit)

    def forward(self, x: Tensor, key_padding_mask: Tensor) -> Tensor:
        for norm_a, attn, norm_m, mlp in zip(
            self.attn_norms, self.attns, self.mlp_norms, self.mlps
        ):
            x = x + attn(norm_a(x), key_padding_mask)
            x = x + mlp(norm_m(x))
        return self.ln(x)


class ContinuousPositionalEncoding(nn.Module):
    """Sinusoidal encoding of a real-valued timestamp."""

    div_term: Tensor

    def __init__(self, d_model: int):
        super().__init__()
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term, persistent=False)

    def forward(self, timestamps: Tensor) -> Tensor:
        position = timestamps.unsqueeze(-1).float()
        angles = position * self.div_term
        encoding = torch.zeros(
            *timestamps.shape, self.div_term.size(0) * 2,
            device=timestamps.device, dtype=timestamps.dtype,
        )
        encoding[..., 0::2] = torch.sin(angles)
        encoding[..., 1::2] = torch.cos(angles)
        return encoding


class CausalTimeAttention(nn.Module):
    """Causal day-level attention with additive continuous-time conditioning.

    Queries are built from the *target* day's timestamp and keys/values from the
    history day's, so a position asks "what does this agent do at time t?" rather
    than only "what came before?".
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.time_enc = ContinuousPositionalEncoding(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: Tensor, q_ts: Tensor, k_ts: Tensor) -> Tensor:
        b, t, _ = x.shape
        x_q = x + self.time_enc(q_ts)
        x_k = x + self.time_enc(k_ts)

        def heads(proj, source):
            return proj(source).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        causal = torch.triu(
            torch.full((t, t), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        out = F.scaled_dot_product_attention(
            heads(self.q_proj, x_q), heads(self.k_proj, x_k), heads(self.v_proj, x_k),
            attn_mask=causal.unsqueeze(0).unsqueeze(0),
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out_proj(out.transpose(1, 2).contiguous().view(b, t, -1))


class CrossDayBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln_attn = nn.LayerNorm(d_model)
        self.attn = CausalTimeAttention(d_model, n_heads, dropout)
        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, q_ts: Tensor, k_ts: Tensor) -> Tensor:
        x = x + self.attn(self.ln_attn(x), q_ts, k_ts)
        return x + self.mlp(self.ln_mlp(x))


class IBAD(nn.Module):
    """Predicts the next day's template mixture from an agent's day history."""

    def __init__(
        self,
        n_bundles: int,
        n_templates: int,
        n_agents: int,
        d_embed: int = 32,
        d_visit: int = 64,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
        n_visit_layers: int = 4,
        n_visit_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_templates = n_templates

        self.visit_embedder = VisitEmbedder(n_bundles, d_embed, d_visit)
        self.intra_day = IntraDayEncoder(d_visit, n_visit_layers, n_visit_heads, dropout)

        # A learned CLS token prepended to each day's visits; the encoder
        # contextualises it and its output is that day's vector.
        self.cls_token = nn.Parameter(torch.zeros(d_visit))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Identity and behaviour occupy separate halves of the day representation.
        self.agent_embed = nn.Embedding(n_agents, d_model // 2)
        self.seq_proj = nn.Linear(d_visit, d_model // 2)
        # Index 7 is the padding/missing bucket.
        self.dow_embed = nn.Embedding(8, d_model)
        nn.init.trunc_normal_(self.dow_embed.weight, std=0.02)
        self.input_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            CrossDayBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, n_templates)

    def encode_days(
        self, arrival: Tensor, departure: Tensor, bundle_id: Tensor, visit_mask: Tensor
    ) -> Tensor:
        """(B, T, V) visit tensors -> (B, T, d_visit) day vectors."""
        b, t, v = arrival.shape
        flat = b * t
        visits = self.visit_embedder(
            arrival.reshape(flat, v), departure.reshape(flat, v), bundle_id.reshape(flat, v)
        )
        cls = self.cls_token.view(1, 1, -1).expand(flat, 1, -1).to(visits.dtype)
        visits = torch.cat([cls, visits], dim=1)

        mask = visit_mask.reshape(flat, v)
        cls_mask = torch.ones(flat, 1, dtype=torch.bool, device=visits.device)
        encoded = self.intra_day(visits, torch.cat([cls_mask, mask], dim=1))
        return encoded[:, 0, :].reshape(b, t, -1)

    def forward(
        self,
        arrival: Tensor,
        departure: Tensor,
        bundle_id: Tensor,
        visit_mask: Tensor,
        day_ts_norm: Tensor,
        agent_idx: Tensor,
        dow: Tensor,
    ) -> Tensor:
        """Returns (B, T-1, K) logits; position j forecasts day j+1."""
        daily = self.encode_days(arrival, departure, bundle_id, visit_mask)
        history = daily[:, :-1, :]

        history_ts = day_ts_norm[:, :-1]
        target_ts = day_ts_norm[:, 1:]

        seq_feat = self.seq_proj(history)
        agent_emb = self.agent_embed(agent_idx).unsqueeze(1).expand(-1, seq_feat.size(1), -1)
        h = torch.cat([agent_emb, seq_feat], dim=-1)

        # Calendar signal for the day being forecast, on the query side only, so
        # it says when the forecast day falls without altering the history.
        target_dow = dow[:, 1:]
        h = h + self.dow_embed(torch.where(target_dow < 0, 7, target_dow))
        h = self.input_drop(h)

        for block in self.blocks:
            h = block(h, q_ts=target_ts, k_ts=history_ts)
        return self.output_proj(self.ln_f(h))

    @torch.no_grad()
    def anomaly_scores(
        self,
        arrival: Tensor,
        departure: Tensor,
        bundle_id: Tensor,
        visit_mask: Tensor,
        day_ts_norm: Tensor,
        theta_obs: Tensor,
        agent_idx: Tensor,
        dow: Tensor,
    ) -> Tensor:
        """KL(observed ‖ forecast) for every forecast day: ``(B, T-1)``, NaN when padded.

        The same divergence the model is trained to minimise, so a day scores
        high exactly when the forecast was surprised by it.
        """
        self.eval()
        logits = self.forward(
            arrival, departure, bundle_id, visit_mask, day_ts_norm, agent_idx, dow
        )
        theta = theta_obs[:, 1:]
        kl = F.kl_div(F.log_softmax(logits, dim=-1), theta, reduction="none").sum(-1)
        valid = (theta.sum(-1) > 0) & kl.isfinite()
        return torch.where(valid, kl, torch.full_like(kl, float("nan")))
