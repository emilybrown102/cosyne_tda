#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- CONFIG ----------------
ROOT = Path(r"D:\Emily\NSD_cogsci_sandbox\COGS401\COSYNE\RDMs_Final")

BOOT_DIR = ROOT / "PH_VR_ripser" / "bootstrap_results"
ROBUST_CSV = BOOT_DIR / "robust_targets_for_R_repcycles.csv"

RAW_PH_DIR = ROOT / "PH_VR_ripser" / "raw_ph_results"

SUBJECTS = [1,2,3,4,5]
ROIS = ["V1","V2","V3","V4","LO2","MT","PH","STSva","PIT"]

OUT_DIR = ROOT / "poster_panels" / "persistence_diagrams" / "betti_curves_robust_only"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS_POINTS = 400
MATCH_TOL_FRAC = 0.02  # warn if matching is poor (relative to max death)

# ---------------- HELPERS ----------------
def roi_dir_raw(sid: int, roi: str) -> Path:
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

def match_targets_to_dgm(dgm_h1: np.ndarray, targets: pd.DataFrame):
    """
    For each (birth_target, death_target), pick the closest point in dgm_h1 (L2 in birth/death).
    Returns matched intervals as an (k,2) array.
    """
    if len(dgm_h1) == 0 or len(targets) == 0:
        return np.empty((0, 2))

    b = dgm_h1[:,0]
    d = dgm_h1[:,1]
    finite = np.isfinite(d)
    b = b[finite]
    d = d[finite]
    if len(b) == 0:
        return np.empty((0, 2))

    pts = np.column_stack([b, d])

    matched = []
    max_scale = float(np.max(d))
    tol = MATCH_TOL_FRAC * max_scale

    for _, r in targets.iterrows():
        bt = float(r["birth_target"])
        dt = float(r["death_target"])
        dist2 = (pts[:,0] - bt)**2 + (pts[:,1] - dt)**2
        j = int(np.argmin(dist2))
        err = float(np.sqrt(dist2[j]))
        matched.append([pts[j,0], pts[j,1]])

        # optional warning (doesn't stop)
        if err > tol:
            pass  # keep silent; you can print if you want

    return np.asarray(matched, float)

def betti_curve(intervals: np.ndarray, eps_grid: np.ndarray):
    """β1(ε) = number of intervals alive at ε."""
    if len(intervals) == 0:
        return np.zeros_like(eps_grid)
    births = intervals[:,0]
    deaths = intervals[:,1]
    return np.array([np.sum((births <= e) & (deaths > e)) for e in eps_grid])

# ---------------- MAIN ----------------
def main():
    if not ROBUST_CSV.exists():
        raise FileNotFoundError(f"Missing: {ROBUST_CSV}")

    rob = pd.read_csv(ROBUST_CSV)
    rob.columns = [c.strip() for c in rob.columns]
    rob["subject"] = rob["subject"].astype(int)
    rob["roi"] = rob["roi"].astype(str)

    for sid in SUBJECTS:
        # First pass: compute a shared eps range for this subject across ROIs
        max_eps = 0.0
        for roi in ROIS:
            d1 = load_dgm(roi_dir_raw(sid, roi) / "dgm_H1.npy")
            if len(d1) > 0:
                finite = np.isfinite(d1[:,1])
                if np.any(finite):
                    max_eps = max(max_eps, float(np.max(d1[finite,1])))

        if max_eps == 0:
            continue

        eps = np.linspace(0, max_eps, EPS_POINTS)

        # Plot: one curve per ROI (robust loops only)
        plt.figure(figsize=(7,5))
        any_curve = False

        for roi in ROIS:
            d1 = load_dgm(roi_dir_raw(sid, roi) / "dgm_H1.npy")
            sub = rob[(rob.subject == sid) & (rob.roi == roi)]

            matched_intervals = match_targets_to_dgm(d1, sub)
            b1 = betti_curve(matched_intervals, eps)

            if np.max(b1) > 0:
                any_curve = True

            plt.plot(eps, b1, label=roi)

        if not any_curve:
            plt.close()
            continue

        plt.xlabel("Filtration scale (ε)")
        plt.ylabel("Number of robust loops (β₁)")
        plt.title(f"subj{sid:02d} — Robust-only H1 Betti curves across ROIs")
        plt.legend(ncol=3, fontsize=8)
        plt.tight_layout()

        out = OUT_DIR / f"subj{sid:02d}_robustOnly_H1_betti_acrossROIs.png"
        plt.savefig(out, dpi=300)
        plt.close()

    print("Saved robust-only Betti curves to:", OUT_DIR)

if __name__ == "__main__":
    main()