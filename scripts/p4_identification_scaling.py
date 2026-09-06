"""At what sample size does the moment residual (R1) start to identify the investor state?
Trains Emporion v0 on synthetic pools of growing size (dates x assets) with corrections *unpenalised*
(so R1 is the only teacher) and reports recovery correlations for beliefs, mispricing, aggregate risk
tolerance A and index share phi. The answer decides how much the nowcaster must carry (doc 08 §2)."""
from __future__ import annotations

import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.models.emporion_v0 import EmporionV0
from emporion.solvers import sample_markets, train_residual
from emporion.core.ledger import Ledger


def corr(a, b):
    a = a.flatten() - a.mean(); b = b.flatten() - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="1024x8,8192x8,32768x8,8192x32,32768x32")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--root", default="runs/p4")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    os.makedirs(args.root, exist_ok=True); ledger = Ledger(os.path.join(args.root, "ledger"))
    dev = torch.device(args.device); g = torch.Generator(device="cpu").manual_seed(0)
    results = []
    for cell in args.grid.split(","):
        B, N = map(int, cell.split("x"))
        pool = sample_markets(B=B, N=N, gen=g, s_mis=0.05, feat_noise=0.3)
        pool = {k: ({kk: vv.to(dev) for kk, vv in v.items()} if isinstance(v, dict) and k != "truth" else v) for k, v in pool.items()}
        for k in ("feats", "p_obs", "D_real", "G"):
            pool[k] = pool[k].to(dev)
        tr = pool["truth"]; ev_n = min(1024, B)
        ev = {k: pool[k][:ev_n] for k in ("feats", "p_obs", "D_real", "G")}; mk = {k: v[:ev_n] for k, v in pool["mkt"].items()}
        model = EmporionV0(in_dim=5, d_model=96, n_layers=2, n_heads=4, n_draws=256).double().to(dev)
        def rep():
            with torch.no_grad():
                _, parts, _, out = model.loss(ev["feats"], ev["p_obs"], ev["D_real"], ev["G"], mk, w_pen=0.0, w_mu=0.0)
            st = out["state"]; A_hat = (1 / st.a).sum(1).cpu(); A_true = (1 / tr["state"].a[:ev_n]).sum(1)
            return {"r1": float(parts["r1"]), "corr_mu": corr(out["mu_hat"].cpu(), tr["mu"][:ev_n]),
                    "corr_mis": corr(out["r_mkt"].cpu(), tr["eta"][:ev_n]), "corr_A": corr(A_hat, A_true),
                    "corr_phi": corr(st.phi.cpu(), tr["state"].phi[:ev_n])}
        r0 = rep(); t0 = time.perf_counter()
        # unpenalised: R1 is the only teacher; large batch so the GMM average is less noisy
        bs = min(2048, B)
        opt_steps = args.steps
        train_residual(model, pool, steps=opt_steps, batch=bs, lr=1e-3, gen=None, log_every=opt_steps, verbose=False,
                       balance_every=10 ** 9, w_pen=0.0, w_mu=0.0)
        r1 = rep(); wall = time.perf_counter() - t0
        row = {"dates": B, "assets": N, "moment_obs": B * N * 6, "steps": opt_steps, "batch": bs, "wall_s": wall, "init": r0, "trained": r1}
        results.append(row)
        print(f"[ident] {B:6d}x{N:3d}  init: mis={r0['corr_mis']:+.2f} A={r0['corr_A']:+.2f} phi={r0['corr_phi']:+.2f} R1={r0['r1']:.4f} | "
              f"trained: mis={r1['corr_mis']:+.2f} mu={r1['corr_mu']:+.2f} A={r1['corr_A']:+.2f} phi={r1['corr_phi']:+.2f} R1={r1['r1']:.4f}  {wall:.0f}s", flush=True)
        ledger.append("trial", f"ident:{B}x{N}", {"sharpe_is": r1["corr_A"], **row}, scope="p4-identification-synthetic")
    with open(os.path.join(args.root, "identification_scaling.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("[done]")


if __name__ == "__main__":
    main()
