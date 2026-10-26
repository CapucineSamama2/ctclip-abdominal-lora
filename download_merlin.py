import os

# ── Paths: set these two environment variables (or edit here) ──────────────
ROOT       = os.environ.get("PROJECT_ROOT", ".")          # data, labels, checkpoints, outputs
CTCLIP_DIR = os.environ.get("CTCLIP_DIR", "./CT-CLIP")    # clone of github.com/ibrahimethemhamamci/CT-CLIP
from huggingface_hub import snapshot_download
snapshot_download(repo_id='stanfordmimi/Merlin', local_dir=os.path.join(ROOT, "merlin_weights"))
