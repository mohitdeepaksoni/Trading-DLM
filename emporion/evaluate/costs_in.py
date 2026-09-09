"""Indian cash-equity (delivery) transaction costs, statutory + exchange, as of 2026 (plan/PHASES-BY-MARKET.md Phase IN step 5).
Components per side, in basis points of traded value:
  STT (securities transaction tax)  10.0 bp on BOTH buy and sell for delivery trades (Finance Act; 0.1 %)
  stamp duty                         1.5 bp on the BUY side only (0.015 %, uniform since 2020-07-01)
  exchange transaction charge        ~0.3 bp (NSE 0.00297 %), SEBI turnover fee 0.01 bp, plus 18 % GST on those
  brokerage                          0 for delivery at discount brokers (Zerodha, Groww); a flat cap otherwise — set per broker
  slippage                           half-spread proxy; default 5 bp for the PIT top-200 by traded value, larger for thinner names
Round trip on a top-200 name ≈ 10 + 10 + 1.5 + 0.7 + 2 × 5 ≈ 33 bp; the earlier ladders used a flat 15 bp one-way (30 bp round trip),
so the change is small but the composition is now explicit and ledgered. Unlike the US (no stamp, no STT), the buy side is dearer."""
from __future__ import annotations

import numpy as np

STT_BPS = 10.0            # each side, delivery
STAMP_BUY_BPS = 1.5       # buy side only
EXCH_BPS = 0.3 * 1.18     # exchange + GST
SEBI_BPS = 0.01
DEFAULT_SLIPPAGE_BPS = 5.0


def one_way_cost_bps(brokerage_bps: float = 0.0, slippage_bps: float = DEFAULT_SLIPPAGE_BPS) -> float:
    """Cost charged on every unit of one-way turnover (buy or sell), excluding the buy-only stamp duty."""
    return STT_BPS + EXCH_BPS + SEBI_BPS + brokerage_bps + slippage_bps


def buy_tax_bps() -> float:
    return STAMP_BUY_BPS


def cost_vectors(n: int, brokerage_bps: float = 0.0, slippage_bps=DEFAULT_SLIPPAGE_BPS):
    """Per-asset (one_way_bps, buy_tax_bps) arrays for portfolio_returns_vec; slippage may be a per-asset array."""
    slip = np.broadcast_to(np.asarray(slippage_bps, float), (n,))
    return STT_BPS + EXCH_BPS + SEBI_BPS + brokerage_bps + slip, np.full(n, STAMP_BUY_BPS)


def round_trip_bps(brokerage_bps: float = 0.0, slippage_bps: float = DEFAULT_SLIPPAGE_BPS) -> float:
    return 2 * one_way_cost_bps(brokerage_bps, slippage_bps) + STAMP_BUY_BPS
