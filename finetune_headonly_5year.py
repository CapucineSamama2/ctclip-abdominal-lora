"""
CT-CLIP Head-Only (Linear Probing) Baseline for Multi-Disease 5-Year Prediction
=================================================================================
Loads the ORIGINAL CT-CLIP pretrained weights (not our SSL LoRA/DoRA checkpoint),
freezes the entire encoder, and trains only the classification head.
This is the "minimal adaptation" baseline requested by Moona, to compare against:
  - LoRA/DoRA adaptation (our SSL-pretrained + LoRA/DoRA fine-tuning)
  - Full fine-tuning of the official Merlin model

Same data split, same hyperparameters, same pipeline as our other experiments.
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

sys.path.insert(0, os.path.join(CTCLIP_DIR, "CT_CLIP"))
sys.path.insert(0, os.path.join(CTCLIP_DIR, "transformer_maskgit"))

from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CTCLIP_WEIGHTS  = os.path.join(ROOT, "ct_clip_weights", "models", "CT-CLIP-Related", "CT-CLIP_v2.pt")
MERLIN_DATA_DIR = os.path.join(ROOT, "merlin_data")
LABELS_FILE     = os.path.join(ROOT, "five_years_disease_task.csv")
SAVE_DIR        = os.path.join(ROOT, "finetune_headonly_5year")

# Fine-tuning hyperparameters — same as our other experiments
BATCH_SIZE   = 2
EPOCHS       = 30
LR           = 1e-5
WD           = 1e-4
WARMUP_EPOCHS = 2
NUM_WORKERS  = 4
SAVE_EVERY   = 5

# CT preprocessing — same as our LoRA/DoRA scripts (CT-CLIP settings)
HU_MIN, HU_MAX  = -1000, 1000
TARGET_SPACING  = (1.5, 0.75, 0.75)
TARGET_SHAPE    = (240, 480, 480)

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
    def __init__(self, embed_dim: int = 512, num_diseases: int = 6):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_diseases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ─────────────────────────────────────────────────────────────────────────────
# Model builder — original CT-CLIP, fully frozen (no LoRA)
# ─────────────────────────────────────────────────────────────────────────────

def build_ctclip():
    tokenizer    = BertTokenizer.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True)
    text_encoder = BertModel.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized")
    image_encoder = CTViT(
        dim=512, codebook_size=8192, image_size=480, patch_size=20,
        temporal_patch_size=10, spatial_depth=4, temporal_depth=4,
        dim_head=32, heads=8)
    clip = CTCLIP(
        image_encoder=image_encoder, text_encoder=text_encoder,
        dim_text=768, dim_image=294912, dim_latent=512,
        extra_latent_projection=True,
        use_mlm=False, downsample_image_embeds=False,
        use_all_token_embeds=False)
    return clip, tokenizer


def load_ctclip_weights(clip, path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = clip.load_state_dict(state, strict=False)
    print(f"  Missing: {len(missing)}  Unexpected: {len(unexpected)}")


def freeze_encoder(clip):
    """Freeze the ENTIRE encoder — no LoRA, nothing trainable except the head."""
    for p in clip.parameters():
        p.requires_grad = False

    total     = sum(p.numel() for p in clip.parameters())
    trainable = sum(p.numel() for p in clip.parameters() if p.requires_grad)
    print(f"  Encoder frozen — Trainable in encoder: {trainable/1e6:.4f}M / {total/1e6:.2f}M")
    return clip


def get_image_embedding(clip, vols):
    """Same image encoding path as our LoRA/DoRA scripts."""
    enc_image = clip.visual_transformer(vols, return_encoded_tokens=True)
    enc_image = torch.mean(enc_image, dim=1)
    enc_image = enc_image.view(enc_image.shape[0], -1)
    image_embeds = enc_image[:, :] if enc_image.ndim == 3 else enc_image
    image_latents = clip.to_visual_latent(image_embeds)
    image_latents = F.normalize(image_latents, dim=-1)
    return image_latents


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing (identical to our LoRA/DoRA scripts)
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
        data = data.transpose(2, 0, 1)

        t = torch.tensor(data).unsqueeze(0).unsqueeze(0)
        scale    = [z_sp/TARGET_SPACING[0],
                    xy_sp/TARGET_SPACING[1],
                    xy_sp/TARGET_SPACING[2]]
        new_size = [int(t.shape[2]*scale[0]),
                    int(t.shape[3]*scale[1]),
                    int(t.shape[4]*scale[2])]
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
            return F.pad(x, spec, value=-1.0)

        t = crop_pad(t, TARGET_SHAPE[0], 0)
        t = crop_pad(t, TARGET_SHAPE[1], 1)
        t = crop_pad(t, TARGET_SHAPE[2], 2)
        t = (t / 1000.0).clamp(-1.0, 1.0)
        return t.unsqueeze(0)

    except Exception as e:
        print(f"  [PREPROCESS ERROR] {path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (identical to our other scripts)
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
            return self.__getitem__(random.randint(0, len(self)-1))
        labels = torch.tensor(labels, dtype=torch.float32)
        return vol, labels, sid


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    vols, labels, sids = zip(*batch)
    return torch.stack(vols), torch.stack(labels), list(sids)


# ─────────────────────────────────────────────────────────────────────────────
# Loss / metrics (identical to our other scripts)
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

    print("\n[1] Building CT-CLIP ...")
    clip, _ = build_ctclip()

    print("\n[2] Loading ORIGINAL pretrained CT-CLIP weights ...")
    load_ctclip_weights(clip, CTCLIP_WEIGHTS)

    print("\n[3] Freezing entire encoder (head-only / linear probing) ...")
    clip = freeze_encoder(clip)
    clip = clip.float().to(device)

    print("\n[4] Building classification head ...")
    head = DiseaseHead(embed_dim=512, num_diseases=NUM_DISEASES).to(device)

    total_trainable = (
        sum(p.numel() for p in clip.parameters() if p.requires_grad) +
        sum(p.numel() for p in head.parameters())
    )
    print(f"  Total trainable params: {total_trainable/1e6:.4f}M")

    print("\n[5] Loading datasets ...")
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
        [p for p in clip.parameters() if p.requires_grad] + list(head.parameters())
    )
    optimizer = AdamW(trainable_params, lr=LR, weight_decay=WD, eps=1e-8)

    total_steps  = EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)
    scheduler    = CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=LR * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print(f"\n{'='*60}")
    print(f"  Head-only fine-tuning  {EPOCHS} epochs  —  5-Year Disease Prediction")
    print(f"{'='*60}\n")

    best_val_auroc = 0.0
    global_step    = 0

    for epoch in range(1, EPOCHS + 1):
        clip.eval()  # encoder always in eval mode (frozen, no dropout/BN updates needed)
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
                with torch.no_grad():
                    img_embed = get_image_embedding(clip, vols)  # frozen, no grad needed
                logits = head(img_embed)
                loss   = masked_bce_loss(logits, labels)

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
            n_b    += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                               "lr": f"{optimizer.param_groups[0]['lr']:.1e}"})

        avg_tl = t_loss / max(n_b, 1)

        clip.eval()
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
                    img_embed = get_image_embedding(clip, vols)
                    logits    = head(img_embed)
                    loss      = masked_bce_loss(logits, labels)

                v_loss += loss.item()
                n_v    += 1
                all_logits.append(logits.cpu().float())
                all_labels.append(labels.cpu().float())

        avg_vl  = v_loss / max(n_v, 1)
        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        aurocs  = compute_auroc(all_logits, all_labels)

        print(f"\nEpoch {epoch:03d}  train_loss={avg_tl:.4f}  val_loss={avg_vl:.4f}")
        print("  Val AUROC:")
        for name, auc in aurocs.items():
            print(f"    {name:<35}: {auc:.4f}" if not np.isnan(auc) else f"    {name:<35}: N/A")

        is_best = aurocs["Average"] > best_val_auroc
        if is_best:
            best_val_auroc = aurocs["Average"]

        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS or is_best:
            tag  = "best" if is_best else f"epoch{epoch:03d}"
            ckpt = os.path.join(SAVE_DIR, f"finetune_headonly_{tag}.pt")
            torch.save({
                "epoch":      epoch,
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

    best_ckpt = os.path.join(SAVE_DIR, "finetune_headonly_best.pt")
    state     = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    head.load_state_dict(state["head"])
    clip.eval()
    head.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Test"):
            if batch is None:
                continue
            vols, labels, _ = batch
            vols = vols.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                img_embed = get_image_embedding(clip, vols)
                logits    = head(img_embed)
            all_logits.append(logits.cpu().float())
            all_labels.append(labels.cpu().float())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    test_aurocs = compute_auroc(all_logits, all_labels)

    print("\n  Test AUROC (head-only baseline):")
    for name in DISEASE_NAMES:
        v = test_aurocs.get(name, float("nan"))
        print(f"  {name:<35}: {v:.4f}" if not np.isnan(v) else f"  {name:<35}: N/A")

    avg = test_aurocs.get("Average", float("nan"))
    print(f"\n  Average AUROC: {avg:.4f}" if not np.isnan(avg) else "\n  Average AUROC: N/A")
    print(f"\n  Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    train()
