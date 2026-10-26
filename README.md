# Adapting CT-CLIP to abdominal CT with LoRA / DoRA for 5-year disease prediction

Code for our ISBI 2027 submission. We adapt CT-CLIP (3D ViT + CXR-BERT) to abdominal CT from the
[Merlin dataset](https://stanfordmimi.github.io/merlin/) with parameter-efficient fine-tuning (LoRA / DoRA),
then predict the 5-year onset of six conditions (CVD, IHD, HTN, DM, CKD, OST) with a linear head.

**The data, labels, reports and model weights are not included** (Merlin data requires its own access; see the Merlin page).

## Pipeline

1. **Stage 1 – contrastive re-pretraining (InfoNCE) on Merlin CT + reports**
   - `corrected_ressl_merlin_lora_1.py` – LoRA; `dora_rank32.py` – DoRA (same script, DoRA layers)
   - Rank / alpha / output folder are set with environment variables (defaults: rank 32, alpha 32). The settings we used:
     - LoRA rank 32: `python corrected_ressl_merlin_lora_1.py`
     - LoRA rank 64: `LORA_RANK=64 LORA_ALPHA=64 SAVE_DIR=$PROJECT_ROOT/merlin_ressl_lora_rank64 python corrected_ressl_merlin_lora_1.py`
     - DoRA rank 32: `python dora_rank32.py`
     - DoRA rank 64: `LORA_RANK=64 LORA_ALPHA=64 SAVE_DIR=$PROJECT_ROOT/merlin_ressl_dora_rank64 python dora_rank32.py`
   - If a `ctclip_merlin_lora_best.pt` exists in `SAVE_DIR`, training resumes from it.
2. **Stage 2 – linear head (512 → 6, masked BCE) on top of the adapted encoder**, one script per configuration:
   - `finetune_lora{32,64}_5year.py`, `finetune_lora32_alpha{16,64}_5year.py`, `finetune_lora64_alpha{16,32}_5year.py`, `finetune_lora96_alpha48_5year.py`
   - `finetune_dora{32,64}_5year.py`
   - `finetune_headonly_5year.py` – head only (no adaptation)
   - `finetune_official_merlin_5year.py` – Merlin full fine-tuning baseline (needs `pip install merlin-vlm`)
3. **Evaluation with saved predictions** (test AUROC per disease + sanity check against the reported numbers):
   - `eval_lora64_alpha32_save.py`, `eval_lora32_alpha32_save.py`, `eval_merlin_ft_save.py` → write `*_test_preds.npz` (arrays `logits`, `labels`)
   - `eval_lora{32,64}_alpha16.py` – same evaluation for the alpha = 16 runs
4. **Qualitative maps**: `attention_maps.py` (attention), `gradcam_maps.py`, `gradcam_v2.py` (Grad-CAM++, needs `pytorch-grad-cam`), `gradcam_v3.py` (Grad-CAM) – on the LoRA-64 model.
5. **Confidence intervals**: `bootstrap_ci.py` – paired patient-level bootstrap (2,000 resamples, seed 0, 95% percentile intervals; the same resampled patients for all models). Run it in the folder that contains the three `.npz` files.

Each configuration was trained once (single run); the intervals reflect test-set sampling only.

`download_merlin.py` downloads the Merlin model weights from Hugging Face (used by the Merlin baseline).

## Setup

```bash
pip install -r requirements.txt
git clone https://github.com/ibrahimethemhamamci/CT-CLIP.git
export CTCLIP_DIR=$PWD/CT-CLIP        # CT-CLIP code
export PROJECT_ROOT=/path/to/your/data/folder
```

Expected layout of `PROJECT_ROOT`:

```
ct_clip_weights/models/CT-CLIP-Related/CT-CLIP_v2.pt   # CT-CLIP pretrained weights
merlin_data/                                           # Merlin CT volumes, one <scan_id>.nii.gz per scan (raw, as downloaded)
reports_final.xlsx                                     # Merlin reports (Stage 1)
five_years_disease_task.csv                            # 5-year disease labels (Stage 2)
```

There is no separate preprocessing step: the scripts read the raw `.nii.gz` volumes and do the preprocessing on the fly (HU clipping to [-1000, 1000], resampling, resizing).

Checkpoints and predictions are written under `PROJECT_ROOT` as well. Scripts were run on a single 24 GB GPU
(batch size 2, mixed precision for Stage 2).

## Citation

Paper under review (ISBI 2027). Citation to be added.
