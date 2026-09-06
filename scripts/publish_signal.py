"""Publish one pre-registered signal file now (used by the nightly job and for ad-hoc first publications).
    python scripts/publish_signal.py us_stocks_wide runs/us_stocks_wide composite"""
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.ledger import Ledger
from emporion.signals import build_signals, publish
market, root, key = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "primary")
L = Ledger("runs/signals/ledger")
sig = build_signals(market, root, K=200 if not market.startswith("crypto") else 100, strategy_key=key)
rec = publish(sig, "runs/signals", L)
print(f"{sig['market']}: {sig['strategy']} | universe {sig['universe_size']} scored {sig['n_scored']} positions {sig['n_positions']} | sha256 {rec['sha256'][:16]} chain {rec['chain'][:16]} ledger#{rec['ledger_seq']}")
try:
    subprocess.run(["ots", "stamp", rec["path"]], check=True, capture_output=True, timeout=120); print("ots proof written:", os.path.basename(rec["path"]) + ".ots")
except Exception as e:  # noqa: BLE001
    print("ots skipped:", str(e)[:60])
print("chain ok:", L.verify_chain())
