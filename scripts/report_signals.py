"""Trader-facing report from the shadow chain (research/07): one Markdown + one HTML page built ONLY from published
signal files, the chain, the integrity report and the point-in-time stores — today's books, abstention flags, chain and
proof status, and the realised out-of-sample record of every signal since its first stamped file. Serves nothing;
written into runs/signals/ so the nightly runner ships it with the state. Numbers here are live prints, not backtests.
    python scripts/report_signals.py [--signals runs/signals]
"""
import argparse, datetime as dt, glob, html, json, os, re, sys
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.data.store import BitemporalStore

ROOTS = {"us_stocks_wide": "runs/us_stocks_wide", "crypto_full": "runs/crypto_full", "nse": "runs/nse"}
ap = argparse.ArgumentParser(); ap.add_argument("--signals", default="runs/signals"); args = ap.parse_args()
now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
# signal files only: <market>-YYYY-MM-DD.json. Anything else in the folder (integrity-*, DATA-INVENTORY.json) is not a book.
files = sorted(f for f in glob.glob(os.path.join(args.signals, "*.json")) if re.match(r"^[a-z0-9_-]+-\d{4}-\d{2}-\d{2}\.json$", os.path.basename(f)))
sigs = [json.load(open(f)) for f in files]
by = {}
for s in sigs:
    if "market" in s:
        by.setdefault(s["market"], []).append(s)
chain = [l.split() for l in open(os.path.join(args.signals, "chain.txt")).read().splitlines() if l.strip()] if os.path.exists(os.path.join(args.signals, "chain.txt")) else []
integ = sorted(glob.glob(os.path.join(args.signals, "integrity-*.json"))); integ = json.load(open(integ[-1])) if integ else {}
ots = {os.path.basename(f) + ".ots": v for f, v in [(f, integ.get("ots", {}).get(os.path.basename(f) + ".ots", "no proof")) for f in files]}
prices = {}
def price_panel(root):
    if root in prices: return prices[root]
    st = BitemporalStore(os.path.join(root, "store")); b = st.as_of("bars", now, columns=["close"], event_start=now - dt.timedelta(days=400))
    w = b.pivot(index="event_time", on="symbol", values="close").sort("event_time"); prices[root] = w; return w
def realised(sig):
    """Forward returns of the file's book from its last_bar close: 1/5/20 sessions (where the future exists) vs EW universe."""
    root = ROOTS.get(sig["market"].split("-")[0] if not sig["market"].startswith("us_stocks_wide") else "us_stocks_wide")
    if root is None or not sig.get("last_bar"): return {}
    w = price_panel(root); dates = [d.date() for d in w["event_time"].to_list()]; d0 = dt.date.fromisoformat(sig["last_bar"])
    if d0 not in dates: return {}
    i0 = dates.index(d0); syms = [c for c in w.columns if c != "event_time"]; P = w.select(syms).to_numpy().astype(float); si = {s: i for i, s in enumerate(syms)}
    book = {si[r["symbol"]]: r["target_weight"] for r in sig["rows"] if r["target_weight"] > 0 and r["symbol"] in si}; univ = [si[r["symbol"]] for r in sig["rows"] if r["symbol"] in si]
    out = {}
    for h in (1, 5, 20):
        if i0 + h < len(dates):
            r = P[i0 + h] / P[i0] - 1; out[h] = (float(sum(wt * r[j] for j, wt in book.items())), float(np.nanmean(r[univ])), dates[i0 + h].isoformat())
    return out
md = [f"# Emporion shadow chain — live record", "", f"Generated {now:%Y-%m-%d %H:%M} UTC from {len(files)} published files. Every book below was hashed, chained and timestamp-stamped on the day shown, before the returns that follow it existed. Nothing here is a backtest.", ""]
md += [f"**Chain:** {len(chain)} links, integrity {'OK' if integ.get('ok') else 'NOT VERIFIED'} (checked {integ.get('as_of', integ.get('date', '—'))}). **Proofs:** " + ", ".join(f"{k.replace('.json.ots','')}: {v.replace('_',' ')}" for k, v in sorted(ots.items())[-4:]), ""]
for market, lst in by.items():
    lst = sorted(lst, key=lambda s: s["as_of"]); latest = lst[-1]
    md += [f"## {market}", "", f"Strategy: {latest['strategy']}", "", f"Latest file {latest['as_of'][:10]} (last bar {latest.get('last_bar')}): universe {latest['universe_size']}, scored {latest['n_scored']}, positions {latest['n_positions']}, **{'ABSTAIN' if latest['abstain'] else 'active'}**.", ""]
    top = [r for r in latest["rows"] if r["target_weight"] > 0][:15]
    md += ["Top holdings (equal weight): " + ", ".join(f"{r['symbol']}" for r in top) + (" …" if latest["n_positions"] > 15 else ""), ""]
    md += ["| file day | last bar | positions | 1-session book / EW | 5-session book / EW | 20-session book / EW |", "|---|---|---|---|---|---|"]
    rec = {1: [], 5: [], 20: []}
    for s in lst:
        rr = realised(s); cells = []
        for h in (1, 5, 20):
            if h in rr: b, e, _ = rr[h]; rec[h].append((b, e)); cells.append(f"{b*100:+.2f}% / {e*100:+.2f}%")
            else: cells.append("pending")
        md.append(f"| {s['as_of'][:10]} | {s.get('last_bar')} | {s['n_positions']} | " + " | ".join(cells) + " |")
    md.append("")
    for h in (1, 5, 20):
        if rec[h]:
            b = np.array([x[0] for x in rec[h]]); e = np.array([x[1] for x in rec[h]])
            md.append(f"- {h}-session record over {len(b)} files: book mean {b.mean()*100:+.2f} % vs EW {e.mean()*100:+.2f} %; beat EW {int((b > e).sum())}/{len(b)}. " + ("(too few prints to mean anything yet)" if len(b) < 20 else ""))
    md.append("")
md += ["## How to read this", "", "- A file is one day's book for one pre-registered rule; its sha256 is chained to the previous file and stamped into Bitcoin (OpenTimestamps). Verify any file with `ots verify <file>.ots`.", "- Returns are simple close-to-close from the file's last bar, equal-weight book vs the equal-weight point-in-time universe, before costs. Certification (research/03 §10) needs months of prints; until then this page is a record, not a claim.", "- ABSTAIN means fewer than 30 names had a valid score that day and the book is empty by rule.", ""]
text = "\n".join(md)
with open(os.path.join(args.signals, "REPORT.md"), "w") as f: f.write(text)
body = html.escape(text)
rows = []
for line in text.splitlines():
    if line.startswith("# "): rows.append(f"<h1>{html.escape(line[2:])}</h1>")
    elif line.startswith("## "): rows.append(f"<h2>{html.escape(line[3:])}</h2>")
    elif line.startswith("|"):
        cells = [html.escape(c.strip()) for c in line.strip("|").split("|")]
        if set(line.replace("|", "").strip()) <= set("-: "): continue
        tag = "th" if rows and not rows[-1].startswith("<tr") else "td"
        rows.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
    elif line.startswith("- "): rows.append(f"<li>{html.escape(line[2:])}</li>")
    elif line.strip(): rows.append(f"<p>{html.escape(line)}</p>")
htm = "<html><head><meta charset='utf-8'><title>Emporion shadow chain</title><style>body{font-family:system-ui;max-width:1100px;margin:2em auto;color:#111}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px;font-size:13px}th{background:#eee}h2{margin-top:2em}</style></head><body>"
out_rows = []; in_table = False
for r in rows:
    if r.startswith("<tr") and not in_table: out_rows.append("<table>"); in_table = True
    if not r.startswith("<tr") and in_table: out_rows.append("</table>"); in_table = False
    out_rows.append(r)
if in_table: out_rows.append("</table>")
with open(os.path.join(args.signals, "REPORT.html"), "w") as f: f.write(htm + "\n".join(out_rows).replace("**", "") + "</body></html>")
print(f"[report] {len(files)} files, {len(by)} markets -> {args.signals}/REPORT.md + REPORT.html")
