"""
Merlin (official) Full Fine-tuning for Multi-Disease 5-Year Prediction
========================================================================
Replicates Moona's experiment: load Merlin's *general* pretrained weights
(ImageEmbedding=True) and fully fine-tune (no LoRA/DoRA, backbone unfrozen)
on our downstream task, using the exact same pipeline/hyperparameters as
our other finetune_*_5year.py scripts, for a direct comparison.

NOTE ON PREPROCESSING: Merlin's own preprocessing (per the paper, Section 4.2)
resamples to 1.5mm in-plane / 3mm slice thickness and center-crops to
224x224x160 - this differs from our CT-CLIP preprocessing (240x480x480).
This script uses the Merlin-paper preprocessing spec. If results look
unexpectedly poor, this preprocessing mismatch should be the first thing
to double-check against Merlin's actual data loading code.
"""

import os

# ── Paths: set these two environment variables (or edit here) ──────────────
ROOT       = os.environ.get("PROJECT_ROOT", ".")          # data, labels, checkpoints, outputs
CTCLIP_DIR = os.environ.get("CTCLIP_DIR", "./CT-CLIP")    # clone of github.com/ibrahimethemhamamci/CT-CLIP
import sys
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import nibabel as nib
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from merlin import Merlin

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MERLIN_DATA_DIR = os.path.join(ROOT, "merlin_data")
LABELS_FILE     = os.path.join(ROOT, "five_years_disease_task.csv")
SAVE_DIR        = os.path.join(ROOT, "finetune_official_merlin_5year")

# Fine-tuning hyperparameters — same as our other experiments
BATCH_SIZE   = 2
EPOCHS       = 30
LR           = 1e-5
WD           = 1e-4
WARMUP_EPOCHS = 2
NUM_WORKERS  = 4
SAVE_EVERY   = 5

# CT preprocessing — per Merlin paper Section 4.2 (NOT the CT-CLIP settings)
HU_MIN, HU_MAX  = -1000, 1000
TARGET_SPACING  = (3.0, 1.5, 1.5)     # (z, xy, xy) mm
TARGET_SHAPE    = (160, 224, 224)     # (D, H, W)

DISEASE_NAMES = [
    "Cardiovascular Disease (CVD)",
    "Ischemic Heart Disease (IHD)",
    "Hypertension (HTN)",
    "Diabetes Mellitus (DM)",
    "Chronic Kidney Disease (CKD)",
    "Osteoporosis (OST)",
]
NUM_DISEASES = len(DISEASE_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# Classification head
# ─────────────────────────────────────────────────────────────────────────────

class DiseaseHead(nn.Module):
    """Linear classification head for 5-year disease prediction."""
    def __init__(self, embed_dim: int = 2048, num_diseases: int = 6):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_diseases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # (B, 6) logits


# ─────────────────────────────────────────────────────────────────────────────
# Model builder — official Merlin, general pretrained weights, image-only
# ─────────────────────────────────────────────────────────────────────────────

def build_merlin():
    model = Merlin(ImageEmbedding=True)
    # full fine-tuning: everything trainable (unlike our LoRA/DoRA scripts)
    for p in model.parameters():
        p.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params : {total/1e6:.2f}M (all trainable — full fine-tuning)")
    return model


def get_image_embedding(model, vols):
    """Merlin(ImageEmbedding=True) returns a tuple; outputs[0] shape varies:
    could be (B, 2048), (1, B, 2048), or (B, 1, 2048) depending on batch size.
    We handle all cases by squeezing whichever dim (0 or 1) has size 1."""
    outputs = model(vols)
    embed = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    if embed.dim() == 3:
        if embed.shape[0] == 1:
            embed = embed.squeeze(0)
        elif embed.shape[1] == 1:
            embed = embed.squeeze(1)
    return embed


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — Merlin paper spec (224x224x160, 1.5mm in-plane / 3mm z)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_volume(path: str):
    try:
        nii   = nib.load(path)
        data  = nii.get_fdata(dtype=np.float32)
        zooms = nii.header.get_zooms()[:3]

        if data.shape[2] < 20:
            return None

        xy_sp = float(zooms[0])
        z_sp  = float(zooms[2])

        data = np.clip(data, HU_MIN, HU_MAX)
        data = data.transpose(2, 0, 1)  # (D, H, W)

        t = torch.tensor(data).unsqueeze(0).unsqueeze(0)
        scale    = [z_sp / TARGET_SPACING[0],
                    xy_sp / TARGET_SPACING[1],
                    xy_sp / TARGET_SPACING[2]]
        new_size = [max(1, int(t.shape[2] * scale[0])),
                    max(1, int(t.shape[3] * scale[1])),
                    max(1, int(t.shape[4] * scale[2]))]
        t = F.interpolate(t, size=new_size, mode="trilinear", align_corners=False)
        t = t.squeeze(0).squeeze(0)

        def crop_pad(x, target, dim):
            cur = x.shape[dim]
            if cur >= target:
                s = (cur - target) // 2
                sl = [slice(None)] * len(x.shape)
                sl[dim] = slice(s, s + target)
                return x[tuple(sl)]
            pad = target - cur
            pb, pa = pad // 2, pad - pad // 2
            spec = [0, 0] * len(x.shape)
            idx  = len(x.shape) - 1 - dim
            spec[idx*2], spec[idx*2+1] = pb, pa
            return F.pad(x, spec, value=0.0)

        t = crop_pad(t, TARGET_SHAPE[0], 0)
        t = crop_pad(t, TARGET_SHAPE[1], 1)
        t = crop_pad(t, TARGET_SHAPE[2], 2)

        # map HU [-1000, 1000] -> [0, 1] per Merlin paper (not [-1,1] like CT-CLIP)
        t = (t - HU_MIN) / (HU_MAX - HU_MIN)
        t = t.clamp(0.0, 1.0)
        return t.unsqueeze(0)  # (1, D, H, W)

    except Exception as e:
        print(f"  [PREPROCESS ERROR] {path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — identical filtering logic to our other scripts
# ─────────────────────────────────────────────────────────────────────────────

LABEL_COLS = ["cvd", "ihd", "htn", "dm", "ckd", "ost"]

class FiveYearDataset(Dataset):
    def __init__(self, data_dir, labels_file, split):
        self.data_dir = data_dir
        df = pd.read_csv(labels_file)
        df = df[df["merlin_split"] == split].reset_index(drop=True)

        self.samples = []
        missing = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  Scanning {split}"):
            sid  = str(row["study id"]).strip()
            path = os.path.join(data_dir, f"{sid}.nii.gz")
            if os.path.exists(path):
                labels = []
                for col in LABEL_COLS:
                    val = row.get(col, float("nan"))
                    if pd.isna(val) or float(val) not in [0.0, 1.0]:
                        labels.append(float("nan"))
                    else:
                        labels.append(float(val))
                self.samples.append((path, labels, sid))
            else:
                missing += 1
        print(f"  {split}: {len(self.samples)} samples  ({missing} missing)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, labels, sid = self.samples[idx]
        vol = preprocess_volume(path)
        if vol is None:
            return self.__getitem__(random.randint(0, len(self) - 1))
        labels = torch.tensor(labels, dtype=torch.float32)
        return vol, labels, sid


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    vols, labels, sids = zip(*batch)
    return torch.stack(vols), torch.stack(labels), list(sids)


# ─────────────────────────────────────────────────────────────────────────────
# Loss / metrics — identical to our other scripts
# ─────────────────────────────────────────────────────────────────────────────

def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(targets)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits[mask], targets[mask], reduction="mean")


def compute_auroc(all_logits, all_labels):
    probs  = torch.sigmoid(torch.tensor(all_logits)).numpy()
    labels = np.array(all_labels)
    aurocs = {}
    for i, name in enumerate(DISEASE_NAMES):
        mask = ~np.isnan(labels[:, i])
        if mask.sum() < 2 or len(np.unique(labels[mask, i])) < 2:
            aurocs[name] = float("nan")
            continue
        aurocs[name] = roc_auc_score(labels[mask, i], probs[mask, i])
    valid = [v for v in aurocs.values() if not np.isnan(v)]
    aurocs["Average"] = np.mean(valid) if valid else float("nan")
    return aurocs


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    print("\n[1] Building official Merlin (general pretrained weights, image-only) ...")
    model = build_merlin().to(device)

    print("\n[2] Building classification head ...")
    head = DiseaseHead(embed_dim=2048, num_diseases=NUM_DISEASES).to(device)

    total_trainable = (
        sum(p.numel() for p in model.parameters() if p.requires_grad) +
        sum(p.numel() for p in head.parameters())
    )
    print(f"  Total trainable params: {total_trainable/1e6:.2f}M")

    print("\n[3] Loading datasets ...")
    train_ds = FiveYearDataset(MERLIN_DATA_DIR, LABELS_FILE, "train")
    val_ds   = FiveYearDataset(MERLIN_DATA_DIR, LABELS_FILE, "val")
    test_ds  = FiveYearDataset(MERLIN_DATA_DIR, LABELS_FILE, "test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               drop_last=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               collate_fn=collate_fn)

    print(f"\n  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    trainable_params = (
        [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    )
    optimizer = AdamW(trainable_params, lr=LR, weight_decay=WD, eps=1e-8)

    total_steps  = EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)
    scheduler    = CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=LR * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print(f"\n{'='*60}")
    print(f"  Fine-tuning official Merlin — {EPOCHS} epochs — 5-Year Disease Prediction")
    print(f"{'='*60}\n")

    best_val_auroc = 0.0
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        head.train()
        t_loss, n_b = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{EPOCHS} [train]")
        for batch in pbar:
            if batch is None:
                continue
            vols, labels, _ = batch
            vols   = vols.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                img_embed = get_image_embedding(model, vols)
                logits    = head(img_embed)
                loss      = masked_bce_loss(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if global_step < warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = LR * global_step / max(warmup_steps, 1)
            else:
                scheduler.step()

            t_loss += loss.item()
            n_b += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                               "lr": f"{optimizer.param_groups[0]['lr']:.1e}"})

        avg_tl = t_loss / max(n_b, 1)

        model.eval()
        head.eval()
        v_loss, n_v = 0.0, 0
        all_logits, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:03d}/{EPOCHS} [val]  "):
                if batch is None:
                    continue
                vols, labels, _ = batch
                vols   = vols.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    img_embed = get_image_embedding(model, vols)
                    logits    = head(img_embed)
                    loss      = masked_bce_loss(logits, labels)

                v_loss += loss.item()
                n_v += 1
                all_logits.append(logits.cpu().float())
                all_labels.append(labels.cpu().float())

        avg_vl = v_loss / max(n_v, 1)
        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        aurocs = compute_auroc(all_logits, all_labels)

        print(f"\nEpoch {epoch:03d}  train_loss={avg_tl:.4f}  val_loss={avg_vl:.4f}")
        print("  Val AUROC:")
        for name, auc in aurocs.items():
            print(f"    {name:<35}: {auc:.4f}" if not np.isnan(auc) else f"    {name:<35}: N/A")

        is_best = aurocs["Average"] > best_val_auroc
        if is_best:
            best_val_auroc = aurocs["Average"]

        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS or is_best:
            tag  = "best" if is_best else f"epoch{epoch:03d}"
            ckpt = os.path.join(SAVE_DIR, f"finetune_official_merlin_{tag}.pt")
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "head":       head.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "train_loss": avg_tl,
                "val_loss":   avg_vl,
                "val_auroc":  aurocs,
            }, ckpt)
            print(f"  {'[BEST] ' if is_best else ''}Saved: {ckpt}")

    # ── final test evaluation ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Final evaluation on test set")
    print(f"{'='*60}\n")

    best_ckpt = os.path.join(SAVE_DIR, "finetune_official_merlin_best.pt")
    state     = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    head.load_state_dict(state["head"])
    model.eval()
    head.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Test"):
            if batch is None:
                continue
            vols, labels, _ = batch
            vols = vols.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                img_embed = get_image_embedding(model, vols)
                logits    = head(img_embed)
            all_logits.append(logits.cpu().float())
            all_labels.append(labels.cpu().float())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    test_aurocs = compute_auroc(all_logits, all_labels)

    print("\n  Test AUROC (official Merlin, full fine-tuning):")
    for name in DISEASE_NAMES:
        v = test_aurocs.get(name, float("nan"))
        print(f"  {name:<35}: {v:.4f}" if not np.isnan(v) else f"  {name:<35}: N/A")

    avg = test_aurocs.get("Average", float("nan"))
    print(f"\n  Average AUROC: {avg:.4f}" if not np.isnan(avg) else "\n  Average AUROC: N/A")
    print(f"\n  Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    train()
