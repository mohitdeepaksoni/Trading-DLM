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
import os, sys, tarfile, tempfile
from huggingface_hub import HfApi, hf_hub_download

REPO = os.environ.get("EMPORION_DATA_REPO", "deepakdsoni/emporion-data")
STORES = ["runs/us_stocks_wide/store", "runs/crypto_full/store", "runs/nse/store", "runs/nse/xbrl_cache", "runs/nse/pit_union_all.txt"]
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


def _stage(paths):
    staged = tempfile.mkdtemp(); n = 0; mb = 0.0
    for pth in paths:
        if not os.path.exists(pth):
            continue
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
            tp = hf_hub_download(REPO, "shadow/state.tar.gz", repo_type="dataset", token=tok, local_dir=tempfile.mkdtemp())
            with tarfile.open(tp) as t:
                t.extractall(".")
            print("[shadow_sync] pulled base stores from the tarball:", int(os.path.getsize(tp) / 1e6), "MB")
        except Exception as e:  # noqa: BLE001
            print("[shadow_sync] tarball base unavailable:", str(e)[:80])
    if any(f.startswith(PREFIX + "/") for f in remote):
        from huggingface_hub import snapshot_download
        d = snapshot_download(REPO, repo_type="dataset", token=tok, allow_patterns=[PREFIX + "/**"], local_dir=tempfile.mkdtemp())
        print(f"[shadow_sync] overlaid store mirror: {_mirror(os.path.join(d, *PREFIX.split('/')))} files from {PREFIX}/")
    # 2. the record wins over the stores — but NEVER overwrite a longer local record with a shorter remote one. The node is
    #    primary while the quota blocks store pushes, so its local chain is normally ahead of HF; clobbering chain.txt with an
    #    older remote copy would truncate the live record.
    n_local = _chain_len("runs/signals/chain.txt"); n_remote = 0
    if "shadow/signals/chain.txt" in remote:
        try:
            n_remote = _chain_len(hf_hub_download(REPO, "shadow/signals/chain.txt", repo_type="dataset", token=tok, local_dir=tempfile.mkdtemp()))
        except Exception:  # noqa: BLE001
            pass
    if n_local and n_local >= n_remote:
        print(f"[shadow_sync] keeping the local record (chain {n_local} entries >= remote {n_remote}); no record overlay")
        remote = set()
    for local, repo_dir in RECORD:
        if any(f.startswith(repo_dir + "/") for f in remote):
            from huggingface_hub import snapshot_download
            d = snapshot_download(REPO, repo_type="dataset", token=tok, allow_patterns=[repo_dir + "/**"], local_dir=tempfile.mkdtemp())
            print(f"[shadow_sync] pulled record: {_mirror(os.path.join(d, *repo_dir.split('/')), local)} files -> {local}")
elif cmd == "push":
    n_local = _chain_len("runs/signals/chain.txt"); n_remote = 0
    for cand in ("shadow/signals/chain.txt", f"{PREFIX}/runs/signals/chain.txt"):
        try:
            n_remote = max(n_remote, _chain_len(hf_hub_download(REPO, cand, repo_type="dataset", token=tok, local_dir=tempfile.mkdtemp())))
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
