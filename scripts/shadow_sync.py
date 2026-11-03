"""Shadow-chain state <-> private HF dataset (the state store for the GitHub Actions runner).
    python scripts/shadow_sync.py pull     # tarball base + shadow/state/** mirror, then the RECORD overlaid from shadow/signals|paper
    python scripts/shadow_sync.py push     # RECORD first (always), then the big stores (soft-fail)
Token from HF_TOKEN.

TWO TIERS, DECOUPLED (2026-10-16, incident 7). The record — signal files, .ots proofs, chain.txt, reports, the paper book —
is a few hundred KB a night of small JSON/MD/TXT, which git stores inline and HF does **not** charge against the private-storage
quota. The point-in-time stores are parquet, which is LFS-tracked and **is** charged. When the quota filled, the old push did the
stores first, threw, and never reached the signal files, so three nights of a valid, Bitcoin-stamped record never reached HF.
Now the record is pushed first and a store failure cannot cost it; the stores retry every night and resume by themselves once
space exists. Degraded mode is safe: a stale store simply makes `refresh_bars` re-fetch the missing sessions that night, so the
published file still carries the correct `last_bar` — the record stays true, only the archived stores wait.

FILE-LEVEL, NOT A TARBALL (2026-10-13). The stores used to be packed into one ~300-470 MB `state.tar.gz` and uploaded nightly;
an HF repo is git, so 44 versions sat in history (~13 GB) and filled the quota. Mirroring individual files means an unchanged
parquet part is skipped by its hash and a night writes ~10-40 MB. The tarball is still read on pull for older state."""
import atexit, os, shutil, sys, tarfile, tempfile
from huggingface_hub import HfApi, hf_hub_download

# Every download and staging copy used to get its own tempfile.mkdtemp() and none of them were ever removed: seven nights of
# pulls and pushes left 43 GB in /tmp on the node (incident 11). Now there is one work root per invocation, named per purpose
# and deleted on exit whatever happens, so a run's peak footprint is bounded and nothing survives it.
WORK = tempfile.mkdtemp(prefix="shadow_sync-")
atexit.register(lambda: shutil.rmtree(WORK, ignore_errors=True))


def _tmp(name):
    d = os.path.join(WORK, name)
    os.makedirs(d, exist_ok=True)
    return d

REPO = os.environ.get("EMPORION_DATA_REPO", "mohitsatna/emporion-data")   # moved 2026-11-03 from deepakdsoni/emporion-data (quota full)
# the ledgers are parquet, so they travel with the stores; before 2026-11-03 they were in neither tier and every Actions run
# started them empty ("ledgers 1/1" on the runner vs "5/5" on the node) — incident 12
STORES = ["runs/us_stocks_wide/store", "runs/crypto_full/store", "runs/nse/store", "runs/nse/xbrl_cache", "runs/nse/pit_union_all.txt",
          "runs/signals/ledger", "runs/paper/nse/ledger"]
RECORD = [("runs/signals", "shadow/signals"), ("runs/paper", "shadow/paper")]   # small, not LFS-tracked -> always pushable
PREFIX = "shadow/state"
# the ledgers are parquet (LFS-tracked, and so quota-bound); they travel with the stores, never with the record
RECORD_IGNORE = ["ledger/*", "*/ledger/*", "*.parquet", "*.bak"]
IGNORE = ["*.bak", "*.bak-*/*", "_*.bak-*/*", "*.tmp", "**/__pycache__/*"]
tok = os.environ.get("HF_TOKEN"); api = HfApi(token=tok)
cmd = sys.argv[1] if len(sys.argv) > 1 else "pull"


def _chain_len(path):
    return len([l for l in open(path).read().splitlines() if l.strip()]) if os.path.exists(path) else 0


def _mirror(src_dir, dst_root="."):
    n = 0
    for root, _, fs in os.walk(src_dir):
        for f in fs:
            s = os.path.join(root, f); rel = os.path.relpath(s, src_dir); dst = os.path.join(dst_root, rel)
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(s):
                open(dst, "wb").write(open(s, "rb").read())
            n += 1
    return n


# HF allows at most 10,000 files per directory; the NSE XBRL cache holds ~14,000 small files, so the file mirror is refused
# (2026-11-03). Directories listed here travel as ONE tar.gz beside the mirror and are unpacked on pull. The cache changes only
# on the weekly refresh, and HF's chunk-level dedup (Xet) keeps an unchanged archive nearly free to re-upload.
ARCHIVED = {"runs/nse/xbrl_cache"}


def _stage(paths):
    staged = _tmp("stage-stores"); n = 0; mb = 0.0
    for pth in paths:
        if not os.path.exists(pth):
            continue
        if pth in ARCHIVED and os.path.isdir(pth):
            dst = os.path.join(staged, pth + ".tar.gz"); os.makedirs(os.path.dirname(dst), exist_ok=True)
            with tarfile.open(dst, "w:gz", compresslevel=6) as t:
                t.add(pth, arcname=os.path.basename(pth))
            n += 1; mb += os.path.getsize(dst) / 1e6; continue
        if os.path.isfile(pth):
            dst = os.path.join(staged, pth); os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, "wb").write(open(pth, "rb").read()); n += 1; mb += os.path.getsize(pth) / 1e6; continue
        for root, dirs, fs in os.walk(pth):
            dirs[:] = [d for d in dirs if ".bak" not in d and d != "__pycache__"]
            for f in fs:
                if f.endswith((".bak", ".tmp")):
                    continue
                s = os.path.join(root, f); dst = os.path.join(staged, s)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                open(dst, "wb").write(open(s, "rb").read()); n += 1; mb += os.path.getsize(s) / 1e6
    return staged, n, mb


if cmd == "pull":
    remote = set(api.list_repo_files(REPO, repo_type="dataset"))
    # 1. the big stores: tarball FIRST as the complete base, then the file mirror on top. The mirror can be PARTIAL — the
    #    10-13 push landed 75 of ~600 files before the quota stopped it — so it must never be treated as a whole state.
    if "shadow/state.tar.gz" in remote:
        try:
            tp = hf_hub_download(REPO, "shadow/state.tar.gz", repo_type="dataset", token=tok, local_dir=_tmp("stores-tarball"))
            with tarfile.open(tp) as t:
                t.extractall(".")
            print("[shadow_sync] pulled base stores from the tarball:", int(os.path.getsize(tp) / 1e6), "MB")
        except Exception as e:  # noqa: BLE001
            print("[shadow_sync] tarball base unavailable:", str(e)[:80])
    if any(f.startswith(PREFIX + "/") for f in remote):
        from huggingface_hub import snapshot_download
        d = snapshot_download(REPO, repo_type="dataset", token=tok, allow_patterns=[PREFIX + "/**"], local_dir=_tmp("stores-mirror"))
        print(f"[shadow_sync] overlaid store mirror: {_mirror(os.path.join(d, *PREFIX.split('/')))} files from {PREFIX}/")
        for a in ARCHIVED:                                                     # unpack archived directories
            if os.path.exists(a + ".tar.gz"):
                with tarfile.open(a + ".tar.gz") as t:
                    t.extractall(os.path.dirname(a))
                os.remove(a + ".tar.gz"); print(f"[shadow_sync] unpacked {a}")
    # 2. the record wins over the stores — but NEVER overwrite a longer local record with a shorter remote one. The node is
    #    primary while the quota blocks store pushes, so its local chain is normally ahead of HF; clobbering chain.txt with an
    #    older remote copy would truncate the live record.
    n_local = _chain_len("runs/signals/chain.txt"); n_remote = 0
    if "shadow/signals/chain.txt" in remote:
        try:
            n_remote = _chain_len(hf_hub_download(REPO, "shadow/signals/chain.txt", repo_type="dataset", token=tok, local_dir=_tmp("remote-chain")))
        except Exception:  # noqa: BLE001
            pass
    if n_local and n_local >= n_remote:
        print(f"[shadow_sync] keeping the local record (chain {n_local} entries >= remote {n_remote}); no record overlay")
        remote = set()
    for local, repo_dir in RECORD:
        if any(f.startswith(repo_dir + "/") for f in remote):
            from huggingface_hub import snapshot_download
            d = snapshot_download(REPO, repo_type="dataset", token=tok, allow_patterns=[repo_dir + "/**"], local_dir=_tmp("record-" + repo_dir.replace("/", "-")))
            print(f"[shadow_sync] pulled record: {_mirror(os.path.join(d, *repo_dir.split('/')), local)} files -> {local}")
elif cmd == "push":
    n_local = _chain_len("runs/signals/chain.txt"); n_remote = 0
    for cand in ("shadow/signals/chain.txt", f"{PREFIX}/runs/signals/chain.txt"):
        try:
            n_remote = max(n_remote, _chain_len(hf_hub_download(REPO, cand, repo_type="dataset", token=tok, local_dir=_tmp("remote-chain-" + cand.replace("/", "-")))))
        except Exception:  # noqa: BLE001
            pass
    if n_local < n_remote:                                                    # never thin out a thicker remote record
        raise SystemExit(f"[shadow_sync] refusing to push: local chain has {n_local} entries, remote has {n_remote}")
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    for local, repo_dir in RECORD:                                            # 1. THE RECORD — must not be lost to a quota error
        if os.path.isdir(local):
            api.upload_folder(folder_path=local, path_in_repo=repo_dir, repo_id=REPO, repo_type="dataset",
                              commit_message="shadow record (signals, proofs, chain, paper)", ignore_patterns=RECORD_IGNORE)
            print(f"[shadow_sync] pushed record: {local} -> {repo_dir} (chain {n_local} entries)")
    staged, n, mb = _stage(STORES)                                            # 2. the stores — may be quota-blocked
    try:
        api.upload_folder(folder_path=staged, path_in_repo=PREFIX, repo_id=REPO, repo_type="dataset",
                          commit_message="shadow stores (file mirror)", ignore_patterns=IGNORE)
        print(f"[shadow_sync] pushed stores: {n} files, {mb:.0f} MB staged (unchanged files skipped by hash)")
    except Exception as e:  # noqa: BLE001
        msg = " | ".join(str(e).strip().splitlines()[-2:])[:160]
        print(f"[shadow_sync] WARNING stores NOT pushed ({msg}). The record is safe; the stores retry tomorrow. "
              f"If this is the storage quota, see reports/CHAIN-INCIDENTS.md incident 7.")
else:
    raise SystemExit("usage: shadow_sync.py pull|push")
