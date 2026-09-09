"""Shadow-chain state <-> private HF dataset (the state store for the GitHub Actions runner).
    python scripts/shadow_sync.py pull     # download shadow/state.tar and unpack into runs/
    python scripts/shadow_sync.py push     # pack runs/{us_stocks_wide,crypto_full,nse}/store + runs/signals -> shadow/state.tar (one commit)
Token from HF_TOKEN. State = point-in-time stores (bars/facts/...) + the signal files, chain, ledger, .ots proofs."""
import os, sys, tarfile, tempfile
from huggingface_hub import HfApi, hf_hub_download

REPO = os.environ.get("EMPORION_DATA_REPO", "deepakdsoni/emporion-data"); PATHS = ["runs/us_stocks_wide/store", "runs/crypto_full/store", "runs/nse/store", "runs/nse/xbrl_cache", "runs/nse/pit_union_all.txt", "runs/signals", "runs/paper"]
tok = os.environ.get("HF_TOKEN"); api = HfApi(token=tok)
cmd = sys.argv[1] if len(sys.argv) > 1 else "pull"
if cmd == "pull":
    try:
        p = hf_hub_download(REPO, "shadow/state.tar.gz", repo_type="dataset", token=tok, local_dir=tempfile.mkdtemp())
    except Exception:  # noqa: BLE001  (first pull after the switch to gzip)
        p = hf_hub_download(REPO, "shadow/state.tar", repo_type="dataset", token=tok, local_dir=tempfile.mkdtemp())
    with tarfile.open(p) as t:
        t.extractall(".")
    print("[shadow_sync] pulled state:", os.path.getsize(p) // 1e6, "MB")
elif cmd == "push":
    out = os.path.join(tempfile.mkdtemp(), "state.tar.gz")
    with tarfile.open(out, "w:gz", compresslevel=6) as t:
        for pth in PATHS:
            if os.path.exists(pth):
                t.add(pth)
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=out, path_in_repo="shadow/state.tar.gz", repo_id=REPO, repo_type="dataset", commit_message="shadow chain state")
    if os.path.isdir("runs/signals"):
        api.upload_folder(folder_path="runs/signals", path_in_repo="shadow/signals", repo_id=REPO, repo_type="dataset", commit_message="shadow signal files", ignore_patterns=["ledger/*", "*.bak"])
    print("[shadow_sync] pushed state:", os.path.getsize(out) // 1e6, "MB")
else:
    raise SystemExit("usage: shadow_sync.py pull|push")
