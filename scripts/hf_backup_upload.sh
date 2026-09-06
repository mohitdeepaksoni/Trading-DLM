#!/bin/bash
# Upload runs/_backup tarballs to the private HF dataset, retrying through the Hub's 5-minute API rate-limit windows.
cd ~/Trading-DLM && . venv/bin/activate
for i in 1 2 3 4 5 6 7 8; do
  python - <<'PY' && { echo "[hf upload done] $(date -u)"; exit 0; }
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
api.upload_folder(repo_id="deepakdsoni/emporion-data", repo_type="dataset", folder_path="runs/_backup", path_in_repo="backup",
                  commit_message="emporion box backup (stores + xbrl cache + ledgers)")
print("uploaded:", [f for f in api.list_repo_files("deepakdsoni/emporion-data", repo_type="dataset") if f.startswith("backup/")])
PY
  echo "[hf upload] attempt $i failed; sleeping 600 s"; sleep 600
done
echo "[hf upload FAILED after 8 attempts] $(date -u)"
