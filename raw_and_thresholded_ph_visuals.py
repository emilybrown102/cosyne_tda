#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- CONFIG ----------------
ROOT = Path(r"D:\Emily\NSD_cogsci_sandbox\COGS401\COSYNE\RDMs_Final")

# ✅ Robust loops live here
BOOT_DIR = ROOT / "PH_VR_ripser" / "bootstrap_results"
ROBUST_CSV = BOOT_DIR / "robust_targets_for_R_repcycles.csv"

# ✅ Raw PH diagrams live here
RAW_PH_DIR = ROOT / "PH_VR_ripser" / "raw_ph_results"

SUBJECTS = [1,2,3,4,5]
ROIS = ["V1","V2","V3","V4","LO2","MT","PH","STSva","PIT"]

OUT_BASE = ROOT / "poster_panels" / "persistence_diagrams"
RAW_DIR = OUT_BASE / "raw"
THR_DIR = OUT_BASE / "thresholded"
RAW_DIR.mkdir(parents=True, exist_ok=True)
THR_DIR.mkdir(parents=True, exist_ok=True)

PERSISTENCE_THRESHOLD_FRAC = 0.05  # same as bootstrap floor

# ---------------- HELPERS ----------------
def roi_dir_raw(sid: int, roi: str) -> Path:
    """Folder containing raw persistence diagram files."""
    return RAW_PH_DIR / f"subj{sid:02d}" / f"ROI-{roi}"

def load_dgm(path: Path):
    if not path.exists():
        return np.empty((0, 2))
    arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr, float)
    if arr.size == 0:
        return np.empty((0, 2))
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    return arr

def plot_pd(ax, dgm, title, threshold=None):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")

    if len(dgm) == 0:
        ax.text(0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes)
        return

    b = dgm[:, 0]
    d = dgm[:, 1]
    finite = np.isfinite(d)
    b = b[finite]
    d = d[finite]

    if len(b) == 0:
        ax.text(0.5, 0.5, "all deaths=inf", ha="center", va="center", transform=ax.transAxes)
        return

    ax.scatter(b, d, s=12, alpha=0.4)

    mn = float(min(b.min(), d.min()))
    mx = float(max(b.max(), d.max()))
    ax.plot([mn, mx], [mn, mx], linewidth=1)

    if threshold is not None:
        x = np.linspace(mn, mx, 200)
        ax.plot(x, x + threshold, linestyle=":", linewidth=2)

# ---------------- MAIN ----------------
def main():
    if not ROBUST_CSV.exists():
        raise RuntimeError(f"Missing robust targets CSV: {ROBUST_CSV}")

    rob = pd.read_csv(ROBUST_CSV)
    rob.columns = [c.strip() for c in rob.columns]

    required = {"subject", "roi", "cluster_id", "hit_rate", "birth_target", "death_target"}
    missing = required - set(rob.columns)
    if missing:
        raise RuntimeError(f"Robust targets CSV missing columns {missing}")

    # ensure consistent dtype for filtering
    rob["subject"] = rob["subject"].astype(int)
    rob["roi"] = rob["roi"].astype(str)

    for sid in SUBJECTS:
        for roi in ROIS:
            roi_path = roi_dir_raw(sid, roi)
            if not roi_path.exists():
                continue

            d0 = load_dgm(roi_path / "dgm_H0.npy")
            d1 = load_dgm(roi_path / "dgm_H1.npy")
            d2 = load_dgm(roi_path / "dgm_H2.npy")

            # ---------- RAW PLOTS ----------
            for dim, dgm in zip([0, 1, 2], [d0, d1, d2]):
                if len(dgm) == 0:
                    continue
                fig, ax = plt.subplots(figsize=(5, 5))
                plot_pd(ax, dgm, f"subj{sid:02d} {roi} H{dim} (raw)")
                out = RAW_DIR / f"subj{sid:02d}_{roi}_H{dim}.png"
                plt.tight_layout()
                plt.savefig(out, dpi=300)
                plt.close(fig)

            # ---------- THRESHOLDED H1 ----------
            if len(d1) > 0:
                finite = np.isfinite(d1[:, 1])
                if np.any(finite):
                    maxdist = float(np.max(d1[finite, 1]))
                else:
                    # if all deaths inf, fall back to max birth for scaling
                    maxdist = float(np.max(d1[:, 0]))

                threshold = PERSISTENCE_THRESHOLD_FRAC * maxdist

                fig, ax = plt.subplots(figsize=(5, 5))
                plot_pd(ax, d1, f"subj{sid:02d} {roi} H1 (thresholded)", threshold)

                # overlay robust loops (targets)
                sub = rob[(rob.subject == sid) & (rob.roi == roi)]
                if len(sub) > 0:
                    ax.scatter(
                        sub["birth_target"].values,
                        sub["death_target"].values,
                        s=180,
                        marker="*",
                        zorder=5
                    )

                out = THR_DIR / f"subj{sid:02d}_{roi}_H1_thresholded.png"
                plt.tight_layout()
                plt.savefig(out, dpi=300)
                plt.close(fig)

    print("Persistence diagrams saved to:", OUT_BASE)

if __name__ == "__main__":
    main()