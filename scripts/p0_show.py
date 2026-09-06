"""Print the latest P0 certification report per market in one line per baseline."""
import glob, json, sys

markets = sys.argv[1:] or ["synthetic", "us", "crypto", "nse"]
for m in markets:
    files = sorted(glob.glob(f"runs/{m}/p0-report-*.json"))
    if not files:
        print(f"=== {m}: no report yet"); continue
    d = json.load(open(files[-1]))
    print(f"=== {m}: T={d['T']} N={d['N']} cost={d['cost_bps']}bps periods={d['periods']} PBO={d['pbo']:.3f} symbols={d['symbols'][:8]}")
    for n, r in d["reports"].items():
        c, ci, rel = r["cpcv"], r["bootstrap_sharpe_ci"], r["vs_benchmark_ann_excess_ci"]
        vs = f"{rel['lo']*100:+.1f}..{rel['hi']*100:+.1f}%" if rel else "  (benchmark)  "
        fails = [k for k, v in r["checks"].items() if not v]
        print(f"  {n:14s} SR={r['sharpe']:+.2f} ret={r['ann_return']*100:+5.1f}% vol={r['ann_vol']*100:4.1f}% mdd={r['max_drawdown']*100:5.1f}% "
              f"CI[{ci['lo']:+.2f},{ci['hi']:+.2f}] DSR={r['dsr']['dsr']:.3f}(N={r['n_trials']}) "
              f"CPCV p5/p50/p95={c['sharpe_p5']:+.2f}/{c['sharpe_p50']:+.2f}/{c['sharpe_p95']:+.2f} vsEW={vs} "
              f"-> {'CERTIFIED' if r['certified'] else 'not certified: ' + ','.join(fails)}")
