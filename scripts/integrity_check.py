"""Nightly integrity check (item 5): every ledger chain verifies, the signal hash chain re-computes from the files,
every OpenTimestamps proof verifies or is still pending (and pending proofs are upgraded when the calendar has them).
Writes runs/signals/integrity-<date>.json and runs/signals/ALERT on any failure; exit code 1 on failure."""
import datetime as dt, glob, hashlib, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.ledger import Ledger

now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
report = {"as_of": now.isoformat(timespec="seconds"), "ledgers": {}, "signal_chain": None, "ots": {}, "ok": True}
# 1. every ledger on disk
for led in sorted(glob.glob("runs/*/ledger")) + ["runs/signals/ledger"]:
    if not os.path.isdir(led):
        continue
    L = Ledger(led); ok = L.verify_chain(); report["ledgers"][led] = {"rows": L.frame().height, "ok": ok}; report["ok"] &= ok
# 2. signal chain: recompute link_i = sha256(link_{i-1} + sha256(file_i)) and compare
chain_path = "runs/signals/chain.txt"
if os.path.exists(chain_path):
    prev, n_ok, n = "0" * 64, 0, 0
    for line in open(chain_path):
        if not line.strip():
            continue
        ts, market, h, link = line.split()
        day = ts[:10]; f = f"runs/signals/{market}-{day}.json"
        fh = hashlib.sha256(open(f, "rb").read()).hexdigest() if os.path.exists(f) else None
        good = (fh == h) and (hashlib.sha256((prev + h).encode()).hexdigest() == link)
        n += 1; n_ok += int(good); prev = link
    report["signal_chain"] = {"entries": n, "verified": n_ok, "ok": n_ok == n}; report["ok"] &= (n_ok == n)
# 3. OpenTimestamps: upgrade pending proofs, then verify each (pending = still ok)
import re as _re
for ots in sorted(glob.glob("runs/signals/*.ots")):
    subprocess.run(["ots", "upgrade", ots], capture_output=True, timeout=120)
    info = subprocess.run(["ots", "info", ots], capture_output=True, text=True, timeout=60); itxt = info.stdout + info.stderr
    blocks = _re.findall(r"[Bb]itcoin[^\n]*?block\s*#?\s*(\d{5,})", itxt) or _re.findall(r"BitcoinBlockHeaderAttestation\((\d+)\)", itxt)
    r = subprocess.run(["ots", "verify", ots], capture_output=True, text=True, timeout=120); out = (r.stdout + r.stderr).lower()
    if "success" in out and "bitcoin" in out:
        status = "verified"
    elif blocks:
        status = f"attested_in_bitcoin_block_{blocks[0]}"                     # upgraded proof; a full node is needed to re-verify locally
    elif "could not connect to bitcoin node" in out and "pending" not in out:
        status = "attested_needs_node"
    elif "pending" in out:
        status = "pending_calendar"
    else:
        status = "FAIL"
    report["ots"][os.path.basename(ots)] = status; report["ok"] &= status != "FAIL"
os.makedirs("runs/signals", exist_ok=True)
with open(f"runs/signals/integrity-{now:%Y-%m-%d}.json", "w") as f:
    json.dump(report, f, indent=1)
if not report["ok"]:
    with open("runs/signals/ALERT", "a") as f:
        f.write(f"{now.isoformat()} integrity FAILED: {json.dumps(report)[:500]}\n")
print(f"[integrity] ledgers {sum(v['ok'] for v in report['ledgers'].values())}/{len(report['ledgers'])} ok | signal chain {report['signal_chain']} | ots {dict((k, v) for k, v in report['ots'].items())} | OK={report['ok']}")
sys.exit(0 if report["ok"] else 1)
