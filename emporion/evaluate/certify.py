"""The certification gate: turns a strategy's out-of-sample returns and the
ledger's trial count into the doc-03 checklist.  Writes the ledger rows."""
from __future__ import annotations

import hashlib
import os
import sys
from typing import Callable, Dict, Optional

import numpy as np

from ..core.ledger import Ledger
from ..core.hashing import content_hash


def procedure_hash(name: str, extra: Optional[Dict] = None) -> str:
    """Identity of the PROCEDURE that produced a trial: rung name + the calling script's code + its arguments.
    Re-running the same script unchanged yields the same hash (not a new trial); a code or parameter change does."""
    script = sys.argv[0] if sys.argv and os.path.exists(sys.argv[0]) else ""
    code = hashlib.sha256(open(script, "rb").read()).hexdigest() if script else "interactive"
    return content_hash({"name": name, "code": code, "argv": sys.argv[1:], "extra": {k: v for k, v in (extra or {}).items() if k in ("cost_bps", "h", "threshold", "vol_target", "w_pen", "select", "every")}})
from .bootstrap import bootstrap_ci
from .cpcv import cpcv_paths, path_stats
from .dsr import deflated_sharpe
from .metrics import sharpe, summary


def certify(name: str, T: int, strategy: Callable[[np.ndarray, np.ndarray], np.ndarray],
            full_returns: np.ndarray, ledger: Ledger, scope: str, benchmark_returns: Optional[np.ndarray] = None,
            n_groups: int = 6, n_test: int = 2, horizon: int = 1, embargo: int = 5, periods: int = 252,
            B: int = 2000, pbo: Optional[float] = None, extra: Optional[Dict] = None) -> Dict:
    """Records the trial FIRST (so N counts it even if it fails), then evaluates."""
    full = np.nan_to_num(np.asarray(full_returns, float))
    s = summary(full, periods)
    sr_is = float(s["sharpe"][0])
    proc = procedure_hash(name, extra)
    ledger.append("trial", name, {"sharpe_is": sr_is, "T": T, "ann_return": float(s["ann_return"][0]),
                                  "max_drawdown": float(s["max_drawdown"][0]), "procedure": proc, **(extra or {})}, scope=scope)
    n_trials = ledger.n_trials(scope, distinct=True)                  # distinct procedures: the honest search count
    n_all = ledger.n_trials(scope)                                     # every evaluation: the strict count, reported alongside
    trial_srs = np.array(ledger.trial_sharpes(scope, distinct=True))
    # null dispersion from the OTHER trials (leave-one-out): including the candidate's own Sharpe would let a
    # genuine edge raise its own hurdle (validated in tests/test_gate_validation.py); robust MAD-based scale
    others = trial_srs[:-1] if trial_srs.size > 1 else trial_srs
    if others.size > 1:
        mad = float(np.median(np.abs(others - np.median(others)))) * 1.4826
        var_sr = max(mad ** 2, float(others.var(ddof=1)) * 0.5) / periods   # per-period variance, floor at half the plain variance
    else:
        var_sr = 0.0

    paths = cpcv_paths(T, strategy, n_groups, n_test, horizon, embargo)
    ps = path_stats(paths, periods)
    sr_pp = sr_is / np.sqrt(periods)
    dsr = deflated_sharpe(sr_pp, T, n_trials, var_sr, float(s["skew"][0]), float(s["kurtosis"][0]))
    dsr_strict = deflated_sharpe(sr_pp, T, n_all, var_sr, float(s["skew"][0]), float(s["kurtosis"][0]))
    ci = bootstrap_ci(full, lambda x: sharpe(x, periods), B=B)
    rel = None
    if benchmark_returns is not None:
        diff = full - np.nan_to_num(np.asarray(benchmark_returns, float))
        rel = bootstrap_ci(diff, lambda x: np.asarray(x).mean(axis=0) * periods, B=B)
    checks = {
        "bootstrap_ci_excludes_0": ci["lo"] > 0,
        "dsr_ge_0.95": dsr["dsr"] >= 0.95,
        "cpcv_p5_sharpe_gt_0": float(np.quantile(ps["sharpe"], 0.05)) > 0,
        "pbo_le_0.2": (pbo is not None and pbo <= 0.2),
        "beats_benchmark_ci": (rel is not None and rel["lo"] > 0),
    }
    report = {"name": name, "scope": scope, "n_trials": n_trials, "n_evaluations": n_all, "procedure": proc[:16], "dsr_strict": dsr_strict["dsr"], "sharpe": sr_is,
              "ann_return": float(s["ann_return"][0]), "ann_vol": float(s["ann_vol"][0]),
              "max_drawdown": float(s["max_drawdown"][0]), "hit_rate": float(s["hit_rate"][0]),
              "dsr": dsr, "bootstrap_sharpe_ci": ci, "vs_benchmark_ann_excess_ci": rel,
              "cpcv": {"n_paths": int(paths.shape[0]), "sharpe_p5": float(np.quantile(ps["sharpe"], 0.05)),
                       "sharpe_p50": float(np.median(ps["sharpe"])), "sharpe_p95": float(np.quantile(ps["sharpe"], 0.95)),
                       "mdd_worst": float(ps["max_drawdown"].min())},
              "pbo": pbo, "checks": checks, "certified": all(v for k, v in checks.items() if k != "beats_benchmark_ci" or benchmark_returns is not None)}
    ledger.append("certification", name, report, scope=scope)
    return report
