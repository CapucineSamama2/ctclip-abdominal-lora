"""
CT-CLIP LoRA Re-SSL on Merlin Abdominal CT Dataset
===================================================
Usage:
  cd /mnt/ssd1/VLM_models/CT-CLIP
  conda activate ctclip
  python ressl_merlin_lora.py
"""

import os

# ── Paths: set these two environment variables (or edit here) ──────────────
ROOT       = os.environ.get("PROJECT_ROOT", ".")          # data, labels, checkpoints, outputs
CTCLIP_DIR = os.environ.get("CTCLIP_DIR", "./CT-CLIP")    # clone of github.com/ibrahimethemhamamci/CT-CLIP
import sys
import math
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
MERLIN_REPORTS  = os.path.join(ROOT, "reports_final.xlsx")
SAVE_DIR        = os.environ.get("SAVE_DIR", os.path.join(ROOT, "merlin_ressl_dora_rank32"))

LORA_RANK    = int(os.environ.get("LORA_RANK", 32))
LORA_ALPHA   = int(os.environ.get("LORA_ALPHA", 32))
BATCH_SIZE   = 3
GRAD_ACCUM   = 8        # effective batch = 16
EPOCHS       = 30
LR           = 5e-5
WD           = 1e-4
WARMUP_EPOCHS = 2
NUM_WORKERS  = 4
SAVE_EVERY   = 5
MAX_TEXT_LEN = 256

# CT preprocessing
HU_MIN, HU_MAX  = -1000, 1000
TARGET_SPACING  = (1.5, 0.75, 0.75)   # (z, x, y) mm
TARGET_SHAPE    = (240, 480, 480)      # (D, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# ct_clip.forward() expects text.input_ids and text.attention_mask
# ─────────────────────────────────────────────────────────────────────────────

class TextBatch:
    """Wraps tokeniser output into object ct_clip.forward() expects."""
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.input_ids      = input_ids
        self.attention_mask = attention_mask

    def to(self, device):
        return TextBatch(
            self.input_ids.to(device),
            self.attention_mask.to(device),
        )


# ─────────────────────────────────────────────────────────────────────────────
# self.weight = orig.weight causes double .to(device) traversal → OOM
# register_buffer properly deduplicates during model.to(device)
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
        self.lora_A = nn.Parameter(
            torch.randn(rank, orig.in_features) * 0.01)
        self.lora_B = nn.Parameter(
            torch.zeros(orig.out_features, rank))
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
# InfoNCE Loss
# with extra_latent_projection=True, forward returns 4 latents:
#   text_latents, image_latents            (text→image direction)
#   text_latents_extra, image_latents_extra (image→text direction)
# We use both pairs in the symmetric InfoNCE loss
# ─────────────────────────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE matching CT-CLIP's decoupled contrastive objective.
    Uses primary latents for text→image and extra latents for image→text.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))

    @property
    def temperature(self):
        return self.log_temp.exp().clamp(min=0.01, max=2.0)

    def forward(
        self,
        text_lat:       torch.Tensor,
        image_lat:      torch.Tensor,
        text_lat_extra: torch.Tensor,
        image_lat_extra: torch.Tensor,
    ):
        B = text_lat.shape[0]
        labels = torch.arange(B, device=text_lat.device)

        # text→image: primary latents
        sim_t2i = (text_lat @ image_lat.T) / self.temperature
        loss_t2i = F.cross_entropy(sim_t2i, labels)

        # image→text: extra latents (decoupled projection)
        sim_i2t = (image_lat_extra @ text_lat_extra.T) / self.temperature
        loss_i2t = F.cross_entropy(sim_i2t, labels)

        loss = (loss_t2i + loss_i2t) / 2.0

        with torch.no_grad():
            acc_t2i = (sim_t2i.argmax(1) == labels).float().mean()
            acc_i2t = (sim_i2t.argmax(1) == labels).float().mean()
            acc = (acc_t2i + acc_i2t) / 2.0

        return loss, acc


# ─────────────────────────────────────────────────────────────────────────────
# Model
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


def load_weights(clip, path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = clip.load_state_dict(state, strict=False)
    print(f"  Missing: {len(missing)}  Unexpected: {len(unexpected)}")


def freeze_and_lora(clip):
    # freeze everything
    for p in clip.parameters():
        p.requires_grad = False

    # LoRA on spatial + temporal attention
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

    # unfreeze patch embeddings — to_patch_emb only (NOT first_frame variant)
    # to_patch_emb_first_frame only activates on first slice edge case
    # and receives no gradient with standard inputs — keep it frozen
    for name, p in clip.visual_transformer.named_parameters():
        if "to_patch_emb" in name and "first_frame" not in name:
            p.requires_grad = True

    total     = sum(p.numel() for p in clip.parameters())
    trainable = sum(p.numel() for p in clip.parameters() if p.requires_grad)
    print(f"  LoRA layers : {n}")
    print(f"  Trainable   : {trainable/1e6:.4f}M / {total/1e6:.2f}M "
          f"({100*trainable/total:.3f}%)")
    return clip


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
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
        data = data.transpose(2, 0, 1)   # (D, H, W)

        t = torch.tensor(data).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
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
        return t.unsqueeze(0)   # (1, D, H, W)

    except Exception as e:
        print(f"  [PREPROCESS ERROR] {path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MerlinDataset(Dataset):
    def __init__(self, data_dir, reports_xlsx, split, tokenizer,
                 max_text_len=256):
        self.data_dir     = data_dir
        self.tokenizer    = tokenizer
        self.max_text_len = max_text_len

        df = pd.read_excel(reports_xlsx)
        df = df[df["Split"] == split].reset_index(drop=True)

        self.samples = []
        missing = 0
        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=f"  Scanning {split}"):
            sid  = str(row["study id"]).strip()
            path = os.path.join(data_dir, f"{sid}.nii.gz")
            if os.path.exists(path):
                report = (str(row["Findings"])
                          .replace('"', '').replace("'", '')
                          .replace('\n', ' ').replace('\r', ' ').strip())
                if len(report) > 10:
                    self.samples.append((path, report, sid))
            else:
                missing += 1

        print(f"  {split}: {len(self.samples)} pairs  ({missing} missing)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, report, sid = self.samples[idx]
        vol = preprocess_volume(path)
        if vol is None:
            return self.__getitem__(random.randint(0, len(self)-1))

        tokens = self.tokenizer(
            report,
            return_tensors="pt",
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
        )
        return (vol,
                tokens["input_ids"].squeeze(0),
                tokens["attention_mask"].squeeze(0),
                sid)


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    vols, ids, masks, names = zip(*batch)
    return torch.stack(vols), torch.stack(ids), torch.stack(masks), list(names)


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

    # ── build + load ──────────────────────────────────────────────────────────
    print("\n[1] Building model ...")
    clip, tokenizer = build_ctclip()

    print("\n[2] Loading pretrained weights ...")
    load_weights(clip, CTCLIP_WEIGHTS)

    print("\n[3] Applying LoRA ...")
    clip = freeze_and_lora(clip)

    # ── fp32 throughout ───────────────────────────────────────────────────────
    # Mixed fp16/fp32 causes dtype collisions in rel_pos_bias and conv layers
    # fp32 model = 1.74GB — easily fits on 24GB RTX 4090
    # AMP autocast still used on activations during forward/backward
    print("\n[4] Setting fp32 precision ...")
    clip = clip.float()

    clip = clip.to(device)
    torch.cuda.synchronize() if device.type == "cuda" else None
    print(f"  Model on {device}  "
          f"({torch.cuda.memory_allocated()/1e9:.2f}GB allocated)"
          if device.type == "cuda" else f"  Model on CPU")

    # ── loss ──────────────────────────────────────────────────────────────────
    state    = torch.load(CTCLIP_WEIGHTS, map_location="cpu", weights_only=False)
    init_temp = float(state.get("temperature", torch.tensor(0.07)))
    criterion = InfoNCELoss(temperature=init_temp).to(device)
    print(f"\n  InfoNCE temperature (pretrained): {init_temp:.4f}")

    # ── datasets ──────────────────────────────────────────────────────────────
    print("\n[5] Loading Merlin dataset ...")
    train_ds = MerlinDataset(MERLIN_DATA_DIR, MERLIN_REPORTS, "train",
                             tokenizer, MAX_TEXT_LEN)
    val_ds   = MerlinDataset(MERLIN_DATA_DIR, MERLIN_REPORTS, "val",
                             tokenizer, MAX_TEXT_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn,
                              prefetch_factor=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=False, collate_fn=collate_fn)

    print(f"\n  Train: {len(train_ds)} pairs  {len(train_loader)} batches")
    print(f"  Val  : {len(val_ds)} pairs  {len(val_loader)} batches")
    print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")

    # ── optimiser ─────────────────────────────────────────────────────────────
    trainable = [p for p in clip.parameters() if p.requires_grad]
    trainable += list(criterion.parameters())
    optimizer  = AdamW(trainable, lr=LR, weight_decay=WD, eps=1e-8)

    total_steps  = EPOCHS * math.ceil(len(train_loader) / GRAD_ACCUM)
    warmup_steps = WARMUP_EPOCHS * math.ceil(len(train_loader) / GRAD_ACCUM)
    scheduler    = CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=LR*0.01)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # ── training loop ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Starting training  {EPOCHS} epochs")
    print(f"{'='*60}\n")

    best_val   = float("inf")
    global_step = 0

    for epoch in range(1, EPOCHS + 1):

        # train
        clip.train()
        criterion.train()
        t_loss, t_acc, n_b = 0.0, 0.0, 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch:03d}/{EPOCHS} [train]")

        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue

            vols, input_ids, attn_masks, _ = batch
            vols      = vols.to(device, non_blocking=True)
            text_batch = TextBatch(
                input_ids.to(device, non_blocking=True),
                attn_masks.to(device, non_blocking=True),
            )

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                # Fix 3: unpack all 4 latents
                txt_lat, img_lat, txt_lat_x, img_lat_x = clip(
                    text_batch, vols,
                    device=device,
                    return_latents=True,
                )
                loss, acc = criterion(txt_lat, img_lat, txt_lat_x, img_lat_x)
                loss = loss / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                global_step += 1
                if global_step < warmup_steps:
                    for pg in optimizer.param_groups:
                        pg["lr"] = LR * global_step / max(warmup_steps, 1)
                else:
                    scheduler.step()

            t_loss += loss.item() * GRAD_ACCUM
            t_acc  += acc.item()
            n_b    += 1

            pbar.set_postfix({
                "loss": f"{loss.item()*GRAD_ACCUM:.4f}",
                "acc":  f"{acc.item():.3f}",
                "temp": f"{criterion.temperature.item():.3f}",
                "lr":   f"{optimizer.param_groups[0]['lr']:.1e}",
            })

        avg_tl = t_loss / max(n_b, 1)
        avg_ta = t_acc  / max(n_b, 1)

        # val
        clip.eval()
        criterion.eval()
        v_loss, v_acc, n_v = 0.0, 0.0, 0

        with torch.no_grad():
            for batch in tqdm(val_loader,
                              desc=f"Epoch {epoch:03d}/{EPOCHS} [val]  "):
                if batch is None:
                    continue
                vols, input_ids, attn_masks, _ = batch
                vols       = vols.to(device, non_blocking=True)
                text_batch = TextBatch(
                    input_ids.to(device, non_blocking=True),
                    attn_masks.to(device, non_blocking=True),
                )
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    txt_lat, img_lat, txt_lat_x, img_lat_x = clip(
                        text_batch, vols, device=device, return_latents=True)
                    loss, acc = criterion(
                        txt_lat, img_lat, txt_lat_x, img_lat_x)
                v_loss += loss.item()
                v_acc  += acc.item()
                n_v    += 1

        avg_vl = v_loss / max(n_v, 1)
        avg_va = v_acc  / max(n_v, 1)

        print(f"\nEpoch {epoch:03d}  "
              f"train_loss={avg_tl:.4f}  train_acc={avg_ta:.3f}  "
              f"val_loss={avg_vl:.4f}  val_acc={avg_va:.3f}  "
              f"temp={criterion.temperature.item():.4f}\n")

        is_best = avg_vl < best_val
        if is_best:
            best_val = avg_vl

        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS or is_best:
            tag  = "best" if is_best else f"epoch{epoch:03d}"
            ckpt = os.path.join(SAVE_DIR, f"ctclip_merlin_lora_{tag}.pt")
            torch.save({
                "epoch":       epoch,
                "global_step": global_step,
                "model":       clip.state_dict(),
                "temperature": criterion.log_temp.data,
                "optimizer":   optimizer.state_dict(),
                "train_loss":  avg_tl,
                "val_loss":    avg_vl,
                "val_acc":     avg_va,
                "lora_rank":   LORA_RANK,
                "lora_alpha":  LORA_ALPHA,
            }, ckpt)
            print(f"  {'[BEST] ' if is_best else ''}Saved: {ckpt}")

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoints: {SAVE_DIR}")


if __name__ == "__main__":
    train()
