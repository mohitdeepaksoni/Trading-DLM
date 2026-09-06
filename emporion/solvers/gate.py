"""Abstention gate (doc 06 §4): deploy a signal only if the model is self-consistent (post-polish
clearing residual small) AND an ensemble agrees; otherwise 'no opinion'.  r_self and r_mkt are
kept separate on purpose."""
from __future__ import annotations

import torch


@torch.no_grad()
def gate(models, feats, p_obs, mkt, mask=None, r_self_tol=1e-8, disagree_tol=0.02):
    outs = [m.forward(feats, p_obs, mkt, mask) for m in models]
    r_mkt = torch.stack([o["r_mkt"] for o in outs])                           # (K,B,N)
    mean, dis = r_mkt.mean(0), r_mkt.std(0) if len(models) > 1 else torch.zeros_like(r_mkt[0])
    p_pol, r_self, it = models[0].polish(outs[0], mkt, mask)
    ok = (r_self <= r_self_tol).unsqueeze(1) & (dis <= disagree_tol)
    return {"r_mkt": mean, "disagreement": dis, "r_self": r_self, "polish_iters": it, "p_polished": p_pol,
            "act": ok, "abstain_rate": float((~ok).double().mean())}
