"""
CT-CLIP DoRA rank 64 Fine-tuning for Multi-Disease 5-Year Prediction
=============================================================
Downstream task: predict onset of 6 chronic diseases within 5 years.
Diseases: CKD, DM, HTN, IHD, CVD, OST

Usage:
  python finetune_merlin_lora_5year.py

Architecture:
  - CT-CLIP image encoder (frozen backbone)
  - LoRA/DoRA weights (trainable, loaded from SSL checkpoint)
  - Linear classification head (trainable, 512 -> 6)
  - Multi-label BCE loss
  - Evaluation: AUROC per disease + average AUROC
"""

import os

# ── Paths: set these two environment variables (or edit here) ──────────────
ROOT       = os.environ.get("PROJECT_ROOT", ".")          # data, labels, checkpoints, outputs
CTCLIP_DIR = os.environ.get("CTCLIP_DIR", "./CT-CLIP")    # clone of github.com/ibrahimethemhamamci/CT-CLIP
import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path

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
# CONFIG — update paths as needed
# ─────────────────────────────────────────────────────────────────────────────
CTCLIP_WEIGHTS  = os.path.join(ROOT, "ct_clip_weights", "models", "CT-CLIP-Related", "CT-CLIP_v2.pt")
SSL_CHECKPOINT  = os.path.join(ROOT, "merlin_ressl_dora_rank64", "ctclip_merlin_lora_best.pt")  # SSL checkpoint to load
MERLIN_DATA_DIR = os.path.join(ROOT, "merlin_data")
LABELS_FILE     = os.path.join(ROOT, "five_years_disease_task.csv")  # CSV with the 5-year disease labels
SAVE_DIR        = os.path.join(ROOT, "finetune_dora64_5year")

# LoRA config — match SSL checkpoint
LORA_RANK    = 64
LORA_ALPHA   = 64

# Fine-tuning hyperparameters (from Merlin paper section 4.5)
BATCH_SIZE   = 2
EPOCHS       = 0
LR           = 1e-5
WD           = 1e-4
WARMUP_EPOCHS = 2
NUM_WORKERS  = 4
SAVE_EVERY   = 5

# CT preprocessing (same as SSL)
HU_MIN, HU_MAX  = -1000, 1000
TARGET_SPACING  = (1.5, 0.75, 0.75)
TARGET_SHAPE    = (240, 480, 480)

# Disease names (same order as Merlin paper)
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
# LoRA layer (same as SSL)
# ─────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    DoRA (Weight-Decomposed Low-Rank Adaptation) of a frozen nn.Linear layer.
    Liu et al., ICML 2024.
    """
    def __init__(self, orig: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.register_buffer("frozen_weight", orig.weight.data.clone())
        self.has_bias = orig.bias is not None
        if self.has_bias:
            self.register_buffer("frozen_bias", orig.bias.data.clone())
        self.lora_A = nn.Parameter(torch.randn(rank, orig.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(orig.out_features, rank))
        self.scale        = alpha / rank
        self.in_features  = orig.in_features
        self.out_features = orig.out_features
        self.rank         = rank
        weight_norm = orig.weight.data.norm(p=2, dim=1, keepdim=True)
        self.magnitude = nn.Parameter(weight_norm.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.frozen_bias if self.has_bias else None
        lora_update = (self.lora_B @ self.lora_A) * self.scale
        adapted_weight = self.frozen_weight + lora_update
        weight_norm = adapted_weight.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        direction = adapted_weight / weight_norm
        out = F.linear(x, self.magnitude * direction, bias)
        return out

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}, dora=True"



# ─────────────────────────────────────────────────────────────────────────────
# Classification head
# ─────────────────────────────────────────────────────────────────────────────

class DiseaseHead(nn.Module):
    """Linear classification head for 5-year disease prediction."""
    def __init__(self, embed_dim: int = 512, num_diseases: int = 6):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_diseases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # (B, 6) logits


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
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


def load_ssl_checkpoint(clip, path):
    """Load SSL checkpoint (LoRA weights already embedded)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = clip.load_state_dict(state, strict=False)
    print(f"  Missing: {len(missing)}  Unexpected: {len(unexpected)}")


def freeze_and_lora(clip):
    """Freeze backbone, apply LoRA to attention layers."""
    for p in clip.parameters():
        p.requires_grad = False

    n = 0
    for tname in ["enc_spatial_transformer", "enc_temporal_transformer"]:
        for layer in getattr(clip.visual_transformer, tname).layers:
            attn = layer[1]
            for attr in ["to_q", "to_kv", "to_out"]:
                if isinstance(getattr(attn, attr), nn.Linear):
                    setattr(attn, attr,
                            LoRALinear(getattr(attn, attr),
                                       rank=LORA_RANK, alpha=LORA_ALPHA))
                    n += 1

    # unfreeze patch embeddings
    for name, p in clip.visual_transformer.named_parameters():
        if "to_patch_emb" in name and "first_frame" not in name:
            p.requires_grad = True

    total     = sum(p.numel() for p in clip.parameters())
    trainable = sum(p.numel() for p in clip.parameters() if p.requires_grad)
    print(f"  LoRA layers : {n}")
    print(f"  Trainable   : {trainable/1e6:.4f}M / {total/1e6:.2f}M")
    return clip




# ─────────────────────────────────────────────────────────────────────────────
# Image embedding extraction (no text encoder needed for fine-tuning)
# ─────────────────────────────────────────────────────────────────────────────

def get_image_embedding(clip, vols):
    """
    Extract image latent from CT-CLIP visual transformer.
    Replicates the image encoding path from ct_clip.py forward():
      1. visual_transformer -> encoded tokens
      2. mean pool + flatten
      3. project to 512-dim latent space
      4. L2 normalize
    """
    enc_image = clip.visual_transformer(vols, return_encoded_tokens=True)
    enc_image = torch.mean(enc_image, dim=1)
    enc_image = enc_image.view(enc_image.shape[0], -1)
    image_embeds = enc_image[:, :] if enc_image.ndim == 3 else enc_image
    image_latents = clip.to_visual_latent(image_embeds)
    image_latents = F.normalize(image_latents, dim=-1)
    return image_latents  # (B, 512)

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing (same as SSL)
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
        return t.unsqueeze(0)  # (1, D, H, W)

    except Exception as e:
        print(f"  [PREPROCESS ERROR] {path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

# CSV columns (from Moona's file):
#   study id, cvd, ihd, htn, dm, ckd, ost, merlin_split
# Values: 0 = healthy, 1 = progressor, 2 = already diagnosed, 3 = censored
# For training: keep only 0 and 1, mask others as NaN
# Use only rows where merlin_split is train/val/test

LABEL_COLS = ["cvd", "ihd", "htn", "dm", "ckd", "ost"]

class FiveYearDataset(Dataset):
    def __init__(self, data_dir, labels_file, split):
        self.data_dir = data_dir

        df = pd.read_csv(labels_file)

        # Keep only rows with the correct merlin_split
        df = df[df["merlin_split"] == split].reset_index(drop=True)

        self.samples = []
        missing = 0
        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=f"  Scanning {split}"):
            sid  = str(row["study id"]).strip()
            path = os.path.join(data_dir, f"{sid}.nii.gz")
            if os.path.exists(path):
                labels = []
                for col in LABEL_COLS:
                    val = row.get(col, float("nan"))
                    # Only keep class 0 (healthy) and class 1 (progressor)
                    # Classes 2 (already diagnosed) and 3 (censored) → NaN
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
        labels = torch.tensor(labels, dtype=torch.float32)  # (6,) may contain NaN
        return vol, labels, sid


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    vols, labels, sids = zip(*batch)
    return torch.stack(vols), torch.stack(labels), list(sids)


# ─────────────────────────────────────────────────────────────────────────────
# Masked BCE Loss (ignores NaN labels)
# ─────────────────────────────────────────────────────────────────────────────

def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCE loss that ignores NaN labels (censored/already diagnosed patients)."""
    mask = ~torch.isnan(targets)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(
        logits[mask], targets[mask], reduction="mean")
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# AUROC evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_auroc(all_logits, all_labels):
    """Compute per-disease and average AUROC. Ignores NaN labels."""
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

    # ── build model ───────────────────────────────────────────────────────────
    print("\n[1] Building CT-CLIP ...")
    clip, _ = build_ctclip()

    print("\n[2] Loading pretrained CT-CLIP weights ...")
    load_ssl_checkpoint(clip, CTCLIP_WEIGHTS)

    print("\n[3] Applying LoRA / DoRA ...")
    clip = freeze_and_lora(clip)

    print("\n[4] Loading SSL checkpoint (LoRA weights) ...")
    load_ssl_checkpoint(clip, SSL_CHECKPOINT)

    clip = clip.float().to(device)

    # ── classification head ───────────────────────────────────────────────────
    print("\n[5] Building classification head ...")
    head = DiseaseHead(embed_dim=512, num_diseases=NUM_DISEASES).to(device)

    total_trainable = (
        sum(p.numel() for p in clip.parameters() if p.requires_grad) +
        sum(p.numel() for p in head.parameters())
    )
    print(f"  Total trainable params: {total_trainable/1e6:.4f}M")

    # ── datasets ──────────────────────────────────────────────────────────────
    print("\n[6] Loading datasets ...")
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

    # ── optimiser ─────────────────────────────────────────────────────────────
    trainable_params = (
        [p for p in clip.parameters() if p.requires_grad] +
        list(head.parameters())
    )
    optimizer = AdamW(trainable_params, lr=LR, weight_decay=WD, eps=1e-8)

    total_steps  = EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EPOCHS * len(train_loader)
    scheduler    = CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=LR * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # ── training loop ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Fine-tuning  {EPOCHS} epochs  —  5-Year Disease Prediction")
    print(f"{'='*60}\n")

    best_val_auroc = 0.0
    global_step    = 0

    for epoch in range(1, EPOCHS + 1):

        # ── train ─────────────────────────────────────────────────────────────
        clip.train()
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
                # Get image embedding from CT-CLIP encoder
                img_embed = get_image_embedding(clip, vols)  # (B, 512)
                logits    = head(img_embed)                   # (B, 6)
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
            n_b    += 1
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr":   f"{optimizer.param_groups[0]['lr']:.1e}",
            })

        avg_tl = t_loss / max(n_b, 1)

        # ── val ───────────────────────────────────────────────────────────────
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

                v_loss     += loss.item()
                n_v        += 1
                all_logits.append(logits.cpu().float())
                all_labels.append(labels.cpu().float())

        avg_vl  = v_loss / max(n_v, 1)
        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        aurocs  = compute_auroc(all_logits, all_labels)

        print(f"\nEpoch {epoch:03d}  train_loss={avg_tl:.4f}  val_loss={avg_vl:.4f}")
        print(f"  Val AUROC:")
        for name, auc in aurocs.items():
            print(f"    {name:<35}: {auc:.4f}" if not np.isnan(auc) else f"    {name:<35}: N/A")

        is_best = aurocs["Average"] > best_val_auroc
        if is_best:
            best_val_auroc = aurocs["Average"]

        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS or is_best:
            tag  = "best" if is_best else f"epoch{epoch:03d}"
            ckpt = os.path.join(SAVE_DIR, f"finetune_dora64_{tag}.pt")
            torch.save({
                "epoch":          epoch,
                "model":          clip.state_dict(),
                "head":           head.state_dict(),
                "optimizer":      optimizer.state_dict(),
                "train_loss":     avg_tl,
                "val_loss":       avg_vl,
                "val_auroc":      aurocs,
                "lora_rank":      LORA_RANK,
                "lora_alpha":     LORA_ALPHA,
            }, ckpt)
            print(f"  {'[BEST] ' if is_best else ''}Saved: {ckpt}")

    # ── final test evaluation ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Final evaluation on test set")
    print(f"{'='*60}\n")

    best_ckpt = os.path.join(SAVE_DIR, "finetune_dora64_best.pt")
    state     = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    clip.load_state_dict(state["model"])
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

    print("\n  Test AUROC (to compare with Merlin paper Table 10):")
    print(f"  {'Disease':<35}  {'Our Model':>10}  {'Swin':>10}  {'Merlin w/o EHR':>14}")
    print(f"  {'-'*75}")

    # Merlin paper Table 10 baselines (100% training data)
    swin_auroc    = {"CVD": 0.54, "IHD": 0.52, "HTN": 0.61, "DM": 0.73, "CKD": 0.55, "OST": 0.60}
    merlin_no_ehr = {"CVD": 0.74, "IHD": 0.74, "HTN": 0.75, "DM": 0.72, "CKD": 0.77, "OST": 0.80}
    disease_keys  = ["CVD", "IHD", "HTN", "DM", "CKD", "OST"]

    for key, name in zip(disease_keys, DISEASE_NAMES):
        our  = test_aurocs.get(name, float("nan"))
        swin = swin_auroc.get(key, float("nan"))
        mno  = merlin_no_ehr.get(key, float("nan"))
        our_str  = f"{our:.4f}"  if not np.isnan(our)  else "N/A"
        print(f"  {name:<35}  {our_str:>10}  {swin:>10.4f}  {mno:>14.4f}")

    avg = test_aurocs.get("Average", float("nan"))
    print(f"\n  {'Average AUROC':<35}  {avg:.4f if not np.isnan(avg) else 'N/A':>10}  {0.59:>10.4f}  {0.75:>14.4f}")
    print(f"\n  Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    train()
