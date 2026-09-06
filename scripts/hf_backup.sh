#!/bin/bash
# Back up the box's runs/ to a PRIVATE HF dataset repo as a handful of tarballs (the XBRL cache alone is 14k files —
# uploading them one by one hits the Hub's 1000-API-calls/5-min limit). Token comes from venv activate.
cd ~/Trading-DLM && . venv/bin/activate
REPO=deepakdsoni/emporion-data; OUT=runs/_backup; mkdir -p $OUT
STAMP=$(date -u +%Y%m%d)
tar -czf $OUT/nse_xbrl_cache-$STAMP.tar.gz -C runs/nse xbrl_cache
tar -cf  $OUT/stores-$STAMP.tar --exclude='_*.bak-*' $(ls -d runs/*/store)
tar -czf $OUT/ledgers_reports-$STAMP.tar.gz --exclude='store' --exclude='xbrl_cache' --exclude='_backup' --exclude='.cache' runs
ls -la $OUT
python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
api.create_repo("deepakdsoni/emporion-data", repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(repo_id="deepakdsoni/emporion-data", repo_type="dataset", folder_path="runs/_backup", path_in_repo="backup",
                  commit_message="emporion box backup (stores + xbrl cache + ledgers)")
print("uploaded:", [f for f in api.list_repo_files("deepakdsoni/emporion-data", repo_type="dataset") if f.startswith("backup/")])
PY
echo "[hf backup done] $(date -u)"
