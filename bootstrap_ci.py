"""Paired patient-level bootstrap for the ISBI 2027 paper.

Usage (files in the same folder as this script, or pass a folder):
    python3 bootstrap_ci.py [folder]

Inputs: lora64_alpha32_test_preds.npz, lora32_alpha32_test_preds.npz,
        merlin_ft_test_preds.npz  (each with arrays "logits" and "labels", shape N x 6;
        masked labels are NaN, as in the training scripts).
All models are resampled with the SAME patient indices (paired bootstrap).
"""
import sys, os
import numpy as np
from sklearn.metrics import roc_auc_score

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "."
B = 2000
SEED = 0
DISEASES = ["CVD", "IHD", "HTN", "DM", "CKD", "OST"]
MODELS = {
    "LoRA-64 a=32": "lora64_alpha32_test_preds.npz",
    "LoRA-32 a=32": "lora32_alpha32_test_preds.npz",
    "Merlin FT":    "merlin_ft_test_preds.npz",
}
EXPECTED_MEAN = {"LoRA-64 a=32": 0.6771, "LoRA-32 a=32": 0.6662, "Merlin FT": 0.6945}


def aurocs(logits, labels, idx=None):
    """Per-disease AUROC (NaN if one class only) on rows idx; same masking as training scripts."""
    if idx is not None:
        logits, labels = logits[idx], labels[idx]
    out = np.full(labels.shape[1], np.nan)
    for i in range(labels.shape[1]):
        m = ~np.isnan(labels[:, i])
        y = labels[m, i]
        if m.sum() < 2 or len(np.unique(y)) < 2:
            continue
        out[i] = roc_auc_score(y, logits[m, i])  # sigmoid is monotone: same AUROC
    return out


def main():
    data = {}
    for name, fn in MODELS.items():
        z = np.load(os.path.join(FOLDER, fn))
        data[name] = (z["logits"].astype(np.float64), z["labels"].astype(np.float64))
    names = list(data)
    n = data[names[0]][0].shape[0]
    # sanity: all files must describe the same patients in the same order
    for nm in names[1:]:
        a, b = data[names[0]][1], data[nm][1]
        if a.shape != b.shape or not np.array_equal(np.nan_to_num(a, nan=-9), np.nan_to_num(b, nan=-9)):
            sys.exit(f"ERROR: labels of {nm} differ from {names[0]} (not the same patients/order).")
    print(f"N test patients = {n}; labels identical across the 3 files: OK")

    point = {nm: aurocs(*data[nm]) for nm in names}
    for nm in names:
        mean = np.nanmean(point[nm])
        flag = "OK" if abs(mean - EXPECTED_MEAN[nm]) < 0.0015 else "MISMATCH"
        print(f"point estimate {nm:14s} mean={mean:.4f} (expected {EXPECTED_MEAN[nm]:.4f}) {flag}")

    rng = np.random.default_rng(SEED)
    boot = {nm: np.zeros((B, 6)) for nm in names}
    for b in range(B):
        idx = rng.integers(0, n, n)
        for nm in names:
            boot[nm][b] = aurocs(*data[nm], idx=idx)

    def ci(x):
        return np.nanpercentile(x, [2.5, 97.5])

    rows = []
    print(f"\n=== 95% bootstrap CI (B={B}) ===")
    print(f"{'model':14s} " + " ".join(f"{d:>17s}" for d in DISEASES + ["Mean"]))
    for nm in names:
        cells = []
        for i in range(6):
            lo, hi = ci(boot[nm][:, i])
            cells.append(f"{point[nm][i]:.3f} [{lo:.3f},{hi:.3f}]")
            rows.append((nm, DISEASES[i], point[nm][i], lo, hi))
        bm = np.nanmean(boot[nm], axis=1)
        lo, hi = ci(bm)
        pm = np.nanmean(point[nm])
        cells.append(f"{pm:.3f} [{lo:.3f},{hi:.3f}]")
        rows.append((nm, "Mean", pm, lo, hi))
        print(f"{nm:14s} " + " ".join(f"{c:>17s}" for c in cells))

    print("\n=== Paired differences (A - B), 95% CI; * = CI excludes 0 ===")
    pairs = [("LoRA-64 a=32", "LoRA-32 a=32"), ("LoRA-64 a=32", "Merlin FT")]
    for a, bnm in pairs:
        print(f"\n{a}  minus  {bnm}")
        for i, d in enumerate(DISEASES + ["Mean"]):
            if d == "Mean":
                diff_b = np.nanmean(boot[a], axis=1) - np.nanmean(boot[bnm], axis=1)
                diff_p = np.nanmean(point[a]) - np.nanmean(point[bnm])
            else:
                diff_b = boot[a][:, i] - boot[bnm][:, i]
                diff_p = point[a][i] - point[bnm][i]
            lo, hi = ci(diff_b)
            star = "*" if (lo > 0 or hi < 0) else ""
            print(f"  {d:5s} {diff_p:+.3f} [{lo:+.3f}, {hi:+.3f}] {star}")
            rows.append((f"{a} - {bnm}", d, diff_p, lo, hi))

    with open(os.path.join(FOLDER, "bootstrap_results.csv"), "w") as f:
        f.write("model,disease,estimate,ci_low,ci_high\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f}\n")
    print("\nSaved bootstrap_results.csv")


if __name__ == "__main__":
    main()
