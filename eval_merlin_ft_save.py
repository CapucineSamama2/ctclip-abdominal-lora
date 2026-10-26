import os

# ── Paths: set these two environment variables (or edit here) ──────────────
ROOT       = os.environ.get("PROJECT_ROOT", ".")          # data, labels, checkpoints, outputs
CTCLIP_DIR = os.environ.get("CTCLIP_DIR", "./CT-CLIP")    # clone of github.com/ibrahimethemhamamci/CT-CLIP
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import finetune_official_merlin_5year as m

device = torch.device("cuda")
model = m.build_merlin().to(device)
head = m.DiseaseHead(embed_dim=2048, num_diseases=m.NUM_DISEASES).to(device)
ds = m.FiveYearDataset(m.MERLIN_DATA_DIR, m.LABELS_FILE, "test")
loader = DataLoader(ds, batch_size=m.BATCH_SIZE, shuffle=False,
                    num_workers=m.NUM_WORKERS, pin_memory=True,
                    collate_fn=m.collate_fn)
ckpt = os.path.join(m.SAVE_DIR, "finetune_official_merlin_best.pt")
state = torch.load(ckpt, map_location="cpu", weights_only=False)
model.load_state_dict(state["model"])
head.load_state_dict(state["head"])
model.eval()
head.eval()

all_logits, all_labels = [], []
with torch.no_grad():
    for batch in tqdm(loader, desc="  Test"):
        if batch is None:
            continue
        vols, labels, _ = batch
        vols = vols.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = head(m.get_image_embedding(model, vols))
        all_logits.append(logits.cpu().float())
        all_labels.append(labels.cpu().float())

all_logits = torch.cat(all_logits, dim=0).numpy()
all_labels = torch.cat(all_labels, dim=0).numpy()
np.savez(os.path.join(ROOT, "merlin_ft_test_preds.npz"),
         logits=all_logits, labels=all_labels)
aur = m.compute_auroc(all_logits, all_labels)
expected = [0.691, 0.715, 0.673, 0.664, 0.677, 0.746]
got = [aur[n] for n in m.DISEASE_NAMES]
print("GUARD shape:", all_logits.shape, all_labels.shape)
print("GUARD got:", [round(float(x), 4) for x in got])
print("GUARD mean of 6:", round(sum(got) / 6, 4))
print("GUARD OK" if all(abs(g - e) < 0.001 for g, e in zip(got, expected)) else "GUARD FAILED")
