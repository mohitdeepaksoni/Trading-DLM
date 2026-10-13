"""Shadow-chain state <-> private HF dataset (the state store for the GitHub Actions runner).
    python scripts/shadow_sync.py pull     # mirror shadow/state/** into runs/  (falls back to the legacy state.tar.gz)
    python scripts/shadow_sync.py push     # mirror runs/** -> shadow/state/**  (only changed files create new blobs)
Token from HF_TOKEN. State = point-in-time stores (bars/facts/...) + the signal files, chain, ledgers, .ots proofs, paper book.

FILE-LEVEL, NOT A TARBALL (changed 2026-10-13, incident 7). The job used to pack the whole state into one ~300-470 MB
`state.tar.gz` and upload it every night. An HF dataset is a git repo: each nightly push wrote a brand-new blob and git kept
every one of them, so 44 nights of history held ~13 GB of superseded tarballs and the account's private-storage limit was
reached — the push then failed outright, which would have stalled the live record. Mirroring individual files instead means
an unchanged parquet part is skipped by its hash and only the night's new parts (~10-40 MB) are written. The tarball is still
read on pull for backward compatibility, and `backup/*.tar.gz` snapshots (deliberate, dated, infrequent) stay as they are."""
import os, sys, tarfile, tempfile
from huggingface_hub import HfApi, hf_hub_download

REPO = os.environ.get("EMPORION_DATA_REPO", "deepakdsoni/emporion-data")
PATHS = ["runs/us_stocks_wide/store", "runs/crypto_full/store", "runs/nse/store", "runs/nse/xbrl_cache",
         "runs/nse/pit_union_all.txt", "runs/signals", "runs/paper"]
PREFIX = "shadow/state"          # mirror root inside the dataset
IGNORE = ["*.bak", "*.bak-*/*", "_*.bak-*/*", "*.tmp", "**/__pycache__/*"]
tok = os.environ.get("HF_TOKEN"); api = HfApi(token=tok)
cmd = sys.argv[1] if len(sys.argv) > 1 else "pull"


def _chain_len(path):
    return len([l for l in open(path).read().splitlines() if l.strip()]) if os.path.exists(path) else 0


if cmd == "pull":
    remote = set(api.list_repo_files(REPO, repo_type="dataset"))
    mirrored = [f for f in remote if f.startswith(PREFIX + "/")]
    if mirrored:
        from huggingface_hub import snapshot_download
        d = snapshot_download(REPO, repo_type="dataset", token=tok, allow_patterns=[PREFIX + "/**"], local_dir=tempfile.mkdtemp())
        src = os.path.join(d, *PREFIX.split("/")); n = 0
        for root, _, fs in os.walk(src):
            for f in fs:
                s = os.path.join(root, f); rel = os.path.relpath(s, src); dst = os.path.join(".", rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(s):
                    open(dst, "wb").write(open(s, "rb").read())
                n += 1
        print(f"[shadow_sync] pulled state: {n} files from {PREFIX}/")
    else:                                                     # legacy: one tarball
        try:
            p = hf_hub_download(REPO, "shadow/state.tar.gz", repo_type="dataset", token=tok, local_dir=tempfile.mkdtemp())
        except Exception:  # noqa: BLE001
            p = hf_hub_download(REPO, "shadow/state.tar", repo_type="dataset", token=tok, local_dir=tempfile.mkdtemp())
        with tarfile.open(p) as t:
            t.extractall(".")
        print("[shadow_sync] pulled legacy tarball state:", os.path.getsize(p) // 1e6, "MB")
elif cmd == "push":
    # never overwrite a thicker remote record with a thinner local one (a failed pull + the always() push step would)
    n_local = _chain_len("runs/signals/chain.txt")
    try:
        rp = hf_hub_download(REPO, f"{PREFIX}/runs/signals/chain.txt", repo_type="dataset", token=tok, local_dir=tempfile.mkdtemp())
        n_remote = _chain_len(rp)
    except Exception:  # noqa: BLE001
        try:
            rp = hf_hub_download(REPO, "shadow/signals/chain.txt", repo_type="dataset", token=tok, local_dir=tempfile.mkdtemp())
            n_remote = _chain_len(rp)
        except Exception:  # noqa: BLE001
            n_remote = 0
    if n_local < n_remote:
        raise SystemExit(f"[shadow_sync] refusing to push: local chain has {n_local} entries, remote has {n_remote}")
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    staged = tempfile.mkdtemp(); n = 0; mb = 0.0
    for pth in PATHS:                                         # stage the mirror so one commit carries the whole state
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
    api.upload_folder(folder_path=staged, path_in_repo=PREFIX, repo_id=REPO, repo_type="dataset",
                      commit_message="shadow chain state (file mirror)", ignore_patterns=IGNORE)
    # the human-readable copies stay where the reports and links already point
    if os.path.isdir("runs/signals"):
        api.upload_folder(folder_path="runs/signals", path_in_repo="shadow/signals", repo_id=REPO, repo_type="dataset",
                          commit_message="shadow signal files", ignore_patterns=["ledger/*", "*.bak"])
    print(f"[shadow_sync] pushed state: {n} files, {mb:.0f} MB staged (unchanged files are skipped by hash)")
else:
    raise SystemExit("usage: shadow_sync.py pull|push")
