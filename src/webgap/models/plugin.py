"""GACA + TRB + ERPR plugin (shared across inserted decoder layers)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from webgap.constants import NUM_REL_TYPES, REL_NONE
from webgap.config import PluginConfig


@dataclass
class GraphBatch:
    """Per-forward structural tensors (padded). Visual + optional DOM streams."""

    assignment: torch.Tensor  # [B, S, K] visual STAR
    rel_idx: torch.Tensor  # [B, K, K]
    k_valid: torch.Tensor  # [B]
    anchor_types: torch.Tensor  # [B, K]
    visual_mask: torch.Tensor  # [B, S]
    perm: torch.Tensor | None = None  # B5: permute *visual* columns only
    assignment_dom: torch.Tensor | None = None
    rel_idx_dom: torch.Tensor | None = None
    k_valid_dom: torch.Tensor | None = None
    anchor_types_dom: torch.Tensor | None = None
    order_ids: torch.Tensor | None = None  # visual reading-order ranks
    order_ids_dom: torch.Tensor | None = None  # document-order ranks


class BottleneckMHA(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
        self.scale = self.head_dim**-0.5

    def _shape(self, x: torch.Tensor, b: int, s: int):
        return x.view(b, s, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ):
        """
        q: [B, Nq, D], kv: [B, Nk, D]
        attn_bias: [B, H, Nq, Nk] additive or [B, 1, Nq, Nk]
        key_mask: [B, Nk] True = keep
        """
        b, nq, _ = q.shape
        nk = kv.shape[1]
        qq = self._shape(self.q(q), b, nq)
        kk = self._shape(self.k(kv), b, nk)
        vv = self._shape(self.v(kv), b, nk)
        scores = torch.matmul(qq, kk.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            scores = scores + attn_bias
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        if self.dropout and self.training:
            attn = F.dropout(attn, p=self.dropout)
        out = torch.matmul(attn, vv).transpose(1, 2).contiguous().view(b, nq, self.dim)
        out = self.o(out)
        if need_weights:
            return out, attn
        return out, None


class WebGAPPlugin(nn.Module):
    def __init__(self, hidden_size: int, cfg: PluginConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = hidden_size
        d = cfg.bottleneck
        self.in_proj = nn.Linear(hidden_size, d, bias=False)
        self.out_proj = nn.Linear(d, hidden_size, bias=False)
        nn.init.zeros_(self.out_proj.weight)
        self.gather = BottleneckMHA(d, cfg.num_heads, cfg.dropout)
        self.anchor_attn = BottleneckMHA(d, cfg.num_heads, cfg.dropout)
        self.scatter = BottleneckMHA(d, cfg.num_heads, cfg.dropout)
        self.anchor_type_emb = nn.Embedding(16, d)
        self.order_emb = nn.Embedding(64, d)
        nn.init.zeros_(self.order_emb.weight)
        # Token-wise mix of visual vs DOM scatter. Positive bias → visual-heavy start.
        self.mix = nn.Linear(d, 1)
        nn.init.zeros_(self.mix.weight)
        nn.init.constant_(self.mix.bias, float(cfg.gate_bias_init))
        # P6 conflict-gate: extra logits from (token order mismatch, graph-level
        # rank correlation). Negative init cancels visual bias when orders reverse.
        self.cg_tok = nn.Linear(1, 1)
        self.cg_graph = nn.Linear(1, 1)
        nn.init.constant_(self.cg_tok.weight, -2.0)
        nn.init.zeros_(self.cg_tok.bias)
        nn.init.constant_(self.cg_graph.weight, -1.0)
        nn.init.zeros_(self.cg_graph.bias)
        self.norm_c = nn.LayerNorm(d)
        self.norm_x = nn.LayerNorm(d)
        # TRB table: [num_rel, heads] then broadcast. Index 0 forced unused via mask.
        self.trb_table = nn.Parameter(torch.zeros(NUM_REL_TYPES, cfg.num_heads))
        nn.init.normal_(self.trb_table, mean=0.0, std=0.02)
        self.trb_scale = nn.Parameter(torch.full((cfg.num_heads,), 0.01))
        # ControlNet-style: only the last projection is zero. A second zero gate
        # would block all plugin gradients (tanh(0)*0).
        self.gate = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.out_proj.weight)
        # reconstruction head for stage-1
        self.rec_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.last_aux: dict[str, torch.Tensor] = {}

    def _trb_bias(self, rel_idx: torch.Tensor) -> torch.Tensor:
        """rel_idx [B,K,K] -> bias [B,H,K,K]. REL_NONE -> 0."""
        table = self.trb_table.to(dtype=self.trb_scale.dtype)
        # [B,K,K,H]
        b = table[rel_idx.clamp(0, NUM_REL_TYPES - 1)]
        b = b * self.trb_scale.view(1, 1, 1, -1)
        b = b.permute(0, 3, 1, 2).contiguous()
        none = rel_idx.eq(REL_NONE)[:, None, :, :]
        b = b.masked_fill(none, 0.0)
        if not self.cfg.use_trb:
            b = torch.zeros_like(b)
        return b

    def _match_assign(self, assign: torch.Tensor, hidden_states: torch.Tensor, k_valid: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.shape
        if assign.shape[1] == seqlen:
            return assign
        if assign.shape[1] == 1:
            return assign
        if seqlen == 1:
            k = assign.shape[-1]
            a = hidden_states.new_ones(bsz, 1, k)
            kv = k_valid.clamp(min=1).view(bsz, 1, 1)
            return a / kv
        if assign.shape[1] > seqlen:
            return assign[:, :seqlen]
        pad = hidden_states.new_zeros(bsz, seqlen - assign.shape[1], assign.shape[-1])
        return torch.cat([assign, pad], dim=1)

    def _run_stream(
        self,
        x: torch.Tensor,
        assign: torch.Tensor,
        rel_idx: torch.Tensor,
        types: torch.Tensor,
        k_valid: torch.Tensor,
        order_ids: torch.Tensor | None,
        perm: torch.Tensor | None,
    ):
        if perm is not None:
            if perm.ndim == 1:
                assign = assign[:, :, perm]
                rel_idx = rel_idx[:, perm][:, :, perm]
                types = types[:, perm]
                if order_ids is not None:
                    order_ids = order_ids[:, perm]
            else:
                bsz = assign.shape[0]
                assign = torch.stack([assign[i][:, perm[i]] for i in range(bsz)], dim=0)
                rel_idx = torch.stack([rel_idx[i][perm[i]][:, perm[i]] for i in range(bsz)], 0)
                types = torch.stack([types[i][perm[i]] for i in range(bsz)], 0)
                if order_ids is not None:
                    order_ids = torch.stack([order_ids[i][perm[i]] for i in range(bsz)], 0)
        pooled = torch.einsum("bsk,bsd->bkd", assign, x)
        types = types.clamp(0, 15)
        c = pooled + self.anchor_type_emb(types)
        if order_ids is not None:
            c = c + self.order_emb(order_ids.clamp(0, 63))
        c = self.norm_c(c)
        key_keep = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
        c2, _ = self.gather(c, x, key_mask=key_keep)
        c = c + c2
        k = assign.shape[-1]
        k_mask = torch.arange(k, device=x.device)[None, :] < k_valid[:, None]
        bias = self._trb_bias(rel_idx)
        c3, _ = self.anchor_attn(c, c, attn_bias=bias, key_mask=k_mask)
        c = c + c3
        x2, attn = self.scatter(self.norm_x(x), c, key_mask=k_mask, need_weights=True)
        return x2, attn

    def _order_conflict(
        self,
        vis_assign: torch.Tensor,
        dom_assign: torch.Tensor,
        order_ids: torch.Tensor | None,
        order_ids_dom: torch.Tensor | None,
        k_valid: torch.Tensor,
        k_valid_dom: torch.Tensor,
        visual_mask: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Token and graph conflict in [0, 1] from STAR assignment × order ranks.

        Visual and DOM columns are different node sets, so relation matrices are
        not aligned. Expected reading-order vs document-order rank per token is.
        """
        bsz, seqlen, _ = vis_assign.shape
        zero_tok = hidden_states.new_zeros(bsz, seqlen)
        zero_g = hidden_states.new_zeros(bsz)
        if order_ids is None or order_ids_dom is None:
            return zero_tok, zero_g
        ov = order_ids.to(dtype=vis_assign.dtype)
        od = order_ids_dom.to(dtype=dom_assign.dtype)
        kv = vis_assign.shape[-1]
        kd = dom_assign.shape[-1]
        if ov.shape[-1] != kv:
            if ov.shape[-1] > kv:
                ov = ov[:, :kv]
            else:
                ov = torch.nn.functional.pad(ov, (0, kv - ov.shape[-1]))
        if od.shape[-1] != kd:
            if od.shape[-1] > kd:
                od = od[:, :kd]
            else:
                od = torch.nn.functional.pad(od, (0, kd - od.shape[-1]))
        exp_v = torch.einsum("bsk,bk->bs", vis_assign, ov)
        exp_d = torch.einsum("bsk,bk->bs", dom_assign, od)
        nv = exp_v / k_valid.to(dtype=exp_v.dtype).clamp(min=1).unsqueeze(1)
        nd = exp_d / k_valid_dom.to(dtype=exp_d.dtype).clamp(min=1).unsqueeze(1)
        tok = (nv - nd).abs().clamp(0, 1)
        w = visual_mask.to(dtype=tok.dtype) if visual_mask is not None else tok.new_ones(tok.shape)
        if w.shape[1] != seqlen:
            w = tok.new_ones(tok.shape)
        wsum = w.sum(dim=1, keepdim=True).clamp(min=1)
        ma = (nv * w).sum(dim=1, keepdim=True) / wsum
        mb = (nd * w).sum(dim=1, keepdim=True) / wsum
        ac = (nv - ma) * w
        bc = (nd - mb) * w
        num = (ac * bc).sum(dim=1)
        den = (ac.square().sum(dim=1) * bc.square().sum(dim=1)).clamp(min=1e-8).sqrt()
        corr = (num / den).clamp(-1, 1)
        graph = ((1.0 - corr) * 0.5).clamp(0, 1)
        return tok, graph

    def forward(self, hidden_states: torch.Tensor, gb: GraphBatch) -> torch.Tensor:
        """
        hidden_states: [B, S, H]
        Dual-order: visual STAR (geometry) and DOM STAR (per-node, source order)
        mixed by a token-wise gate. B5 permutes only the visual stream.
        """
        if not self.cfg.use_gaca:
            self.last_aux = {
                "erpr": hidden_states.new_zeros(()),
                "rec": hidden_states.new_zeros(()),
                "gate": hidden_states.new_zeros(()),
                "conflict": hidden_states.new_zeros(()),
            }
            return hidden_states

        mode = getattr(self.cfg, "anchor_mode", "visual") or "visual"
        x = self.in_proj(hidden_states)
        vis_delta = vis_attn = None
        if mode != "dom":
            vis_assign = self._match_assign(gb.assignment, hidden_states, gb.k_valid)
            vis_delta, vis_attn = self._run_stream(
                x,
                vis_assign,
                gb.rel_idx,
                gb.anchor_types,
                gb.k_valid,
                gb.order_ids,
                gb.perm,
            )

        use_dom = mode in ("dom", "dual") and gb.assignment_dom is not None
        if use_dom:
            kv = gb.k_valid_dom if gb.k_valid_dom is not None else gb.k_valid
            types = gb.anchor_types_dom if gb.anchor_types_dom is not None else gb.anchor_types
            rel = gb.rel_idx_dom if gb.rel_idx_dom is not None else gb.rel_idx
            dom_assign = self._match_assign(gb.assignment_dom, hidden_states, kv)
            dom_delta, dom_attn = self._run_stream(
                x, dom_assign, rel, types, kv, gb.order_ids_dom, None
            )
        else:
            dom_delta, dom_attn = vis_delta, vis_attn

        gate = x.new_ones(x.shape[0], x.shape[1], 1)
        tok_c = x.new_zeros(x.shape[0], x.shape[1])
        graph_c = x.new_zeros(x.shape[0])
        if mode == "dom":
            delta = dom_delta
            attn = dom_attn
            gate = gate * 0
        elif mode == "dual" and use_dom:
            logit = self.mix(x)
            if getattr(self.cfg, "conflict_gate", False):
                tok_c, graph_c = self._order_conflict(
                    vis_assign,
                    dom_assign,
                    gb.order_ids,
                    gb.order_ids_dom,
                    gb.k_valid,
                    kv,
                    gb.visual_mask,
                    hidden_states,
                )
                logit = logit + self.cg_tok(tok_c.unsqueeze(-1).to(dtype=logit.dtype))
                logit = logit + self.cg_graph(graph_c.view(-1, 1, 1).expand(-1, logit.shape[1], 1).to(dtype=logit.dtype))
            gate = torch.sigmoid(logit)
            delta = gate * vis_delta + (1.0 - gate) * dom_delta
            attn = vis_attn
        else:
            delta = vis_delta
            attn = vis_attn

        gated = self.out_proj(delta)
        out = hidden_states + gated

        erpr = hidden_states.new_zeros(())
        if attn is not None and self.cfg.use_erpr:
            p = attn.clamp_min(1e-8)
            ent = -(p * p.log()).sum(dim=-1)
            k_eff = gb.k_valid.clamp(min=2).float().log().view(hidden_states.shape[0], 1, 1)
            hnorm = (ent / k_eff).mean()
            erpr = F.relu(self.cfg.erpr_tau - hnorm)

        rec = hidden_states.new_zeros(())
        con = hidden_states.new_zeros(())
        if self.training and getattr(self.cfg, "compute_ssl", False):
            ssl = self.ssl_losses(hidden_states, gb)
            rec, con = ssl["rec"], ssl["con"]

        self.last_aux = {
            "erpr": erpr,
            "rec": rec,
            "con": con,
            "entropy": (1.0 - erpr.detach()) if isinstance(erpr, torch.Tensor) else erpr,
            "gate": gate.mean() if torch.is_tensor(gate) else hidden_states.new_zeros(()),
            "conflict": tok_c.mean() if torch.is_tensor(tok_c) else hidden_states.new_zeros(()),
        }
        return out

    def ssl_losses(self, hidden_states: torch.Tensor, gb: GraphBatch) -> dict[str, torch.Tensor]:
        """Anchor reconstruction + dropout contrastive on pooled anchors."""
        x = self.in_proj(hidden_states)
        assign = gb.assignment
        if assign.shape[1] != x.shape[1]:
            return {"rec": x.new_zeros(()), "con": x.new_zeros(())}
        pooled = torch.einsum("bsk,bsd->bkd", assign, x).detach()
        bsz, k, d = pooled.shape
        mask = torch.rand(bsz, k, device=x.device) < 0.25
        k_mask = torch.arange(k, device=x.device)[None, :] < gb.k_valid[:, None]
        mask = mask & k_mask
        if mask.sum() == 0:
            rec = x.new_zeros(())
        else:
            vis = pooled.masked_fill(mask[:, :, None], 0.0)
            pred = self.rec_head(vis)
            rec = F.mse_loss(pred[mask], pooled[mask])
        # two dropout views of assignment
        drop1 = F.dropout(assign, p=0.2, training=True)
        drop2 = F.dropout(assign, p=0.2, training=True)
        a1 = F.normalize(torch.einsum("bsk,bsd->bkd", drop1, x).mean(dim=1), dim=-1)
        a2 = F.normalize(torch.einsum("bsk,bsd->bkd", drop2, x).mean(dim=1), dim=-1)
        logits = a1 @ a2.transpose(0, 1) / 0.07
        labels = torch.arange(bsz, device=x.device)
        con = F.cross_entropy(logits, labels)
        return {"rec": rec, "con": con}
