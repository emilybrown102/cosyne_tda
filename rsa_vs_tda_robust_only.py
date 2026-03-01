#!/usr/bin/env python
"""
Robust-only TSA vs RSA (hit_rate >= 0.95)

What this does:
- RSA: ROI×ROI similarity = corr(vectorized RDMs)
- Robust-only TSA: ROI×ROI similarity = exp(-dist/scale) where dist is computed
  between *robust-only* H1 persistence diagrams (each diagram is reduced to K points
  corresponding to robust targets for that subj×ROI at hit_rate >= HITRATE_MIN).

Outputs:
- per-subject: side-by-side heatmaps (RSA vs robust-only TSA)
- per-subject: scatter plot of ROI-pairs (RSA vs TSA)
- group average: side-by-side heatmaps + scatter
- optional: list of top disagreements (largest |z(RSA) - z(TSA)|)

Notes:
- This script does NOT rerun bootstrapping. It only filters robust targets.
- It matches each bootstrap target (birth_target, death_target) to the closest raw H1 PD point.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math

# ---------------- CONFIG ----------------
ROOT = Path(r"D:\Emily\NSD_cogsci_sandbox\COGS401\COSYNE\RDMs_Final")

# RSA inputs
RDM_ROOT = ROOT / "Cross_nob_RDMs"
STRICT_TAG = "STRICT2"

# Robust targets + raw persistence diagrams
BOOT_DIR = ROOT / "PH_VR_ripser" / "bootstrap_results"
TARGETS_CSV = BOOT_DIR / "robust_targets_for_R_repcycles.csv"
RAW_PH_DIR = ROOT / "PH_VR_ripser" / "raw_ph_results"

# Subjects/ROIs
SUBJECTS = [1, 2, 3, 4, 5]
ROI_ORDER = ["V1", "V2", "V3", "V4", "LO2", "MT", "PH", "STSva", "PIT"]

# Robust threshold
HITRATE_MIN = 0.95

# TSA diagram distance
TSA_METRIC = "wasserstein"  # "wasserstein" or "bottleneck"
SCALE_MODE = "median"       # "median" or "mean"

# Output
OUT_DIR = ROOT / "poster_panels" / "TSA_RSA_robustOnly_h095"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Optional: persim distances ----------------
try:
    from persim import wasserstein, bottleneck
    HAVE_PERSIM = True
except Exception:
    HAVE_PERSIM = False


# ---------------- Helpers: loading ----------------
def load_rdm(subj: int, roi: str) -> np.ndarray:
    subj_folder = RDM_ROOT / f"subj{subj:02d}"
    npy = subj_folder / f"ROI-{roi}_crossnobis_{STRICT_TAG}_sessionSplit_raw.npy"
    csv = subj_folder / f"ROI-{roi}_crossnobis_{STRICT_TAG}_sessionSplit_raw.csv"

    if npy.exists():
        D = np.load(npy).astype(float)
    elif csv.exists():
        D = pd.read_csv(csv, header=None).values.astype(float)
    else:
        raise FileNotFoundError(f"Missing RDM for subj{subj:02d} ROI-{roi} (npy/csv)")

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"RDM not square for subj{subj:02d} ROI-{roi}: {D.shape}")

    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def vec_upper(D: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(D.shape[0], k=1)
    return D[i, j]


def load_dgm_h1(subj: int, roi: str) -> np.ndarray:
    p = RAW_PH_DIR / f"subj{subj:02d}" / f"ROI-{roi}" / "dgm_H1.npy"
    if not p.exists():
        return np.empty((0, 2), float)
    arr = np.load(p, allow_pickle=True)
    arr = np.asarray(arr, float)
    if arr.size == 0:
        return np.empty((0, 2), float)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    # finite deaths only for matching/distances
    finite = np.isfinite(arr[:, 1])
    arr = arr[finite]
    if arr.size == 0:
        return np.empty((0, 2), float)
    return arr


# ---------------- Helpers: distances/similarity ----------------
def corr_safe(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size != y.size or x.size < 2:
        return np.nan
    sx = np.std(x)
    sy = np.std(y)
    if sx == 0 or sy == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def pd_distance(dgm_a, dgm_b, metric="wasserstein"):
    if HAVE_PERSIM:
        if metric == "bottleneck":
            return float(bottleneck(dgm_a, dgm_b))
        return float(wasserstein(dgm_a, dgm_b, matching=False))
    # Fallback: abs diff in total persistence (not a true PD distance)
    ta = float(np.sum(dgm_a[:, 1] - dgm_a[:, 0])) if dgm_a.size else 0.0
    tb = float(np.sum(dgm_b[:, 1] - dgm_b[:, 0])) if dgm_b.size else 0.0
    return abs(ta - tb)


def dist_to_sim(Dmat: np.ndarray) -> np.ndarray:
    n = Dmat.shape[0]
    off = Dmat[np.triu_indices(n, k=1)]
    scale = np.nanmedian(off) if SCALE_MODE == "median" else np.nanmean(off)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    Smat = np.exp(-Dmat / scale)
    np.fill_diagonal(Smat, 1.0)
    return Smat


# ---------------- Robust-only PD construction ----------------
def build_robust_only_pd(raw_dgm: np.ndarray, targets_bt_dt: np.ndarray) -> np.ndarray:
    """
    For each target (birth_target, death_target), pick the closest raw PD point.
    Returns an array of matched points (K,2). If no raw points or no targets, returns empty.
    """
    if raw_dgm.size == 0 or targets_bt_dt.size == 0:
        return np.empty((0, 2), float)

    raw = raw_dgm
    out_pts = []
    used = set()

    for bt, dt in targets_bt_dt:
        d2 = (raw[:, 0] - bt) ** 2 + (raw[:, 1] - dt) ** 2
        j = int(np.argmin(d2))
        # avoid duplicating the same raw point for multiple targets if possible
        if j in used and len(raw) > 1:
            order = np.argsort(d2)
            for jj in order:
                jj = int(jj)
                if jj not in used:
                    j = jj
                    break
        used.add(j)
        out_pts.append(raw[j])

    if not out_pts:
        return np.empty((0, 2), float)
    return np.vstack(out_pts)


# ---------------- Compute RSA / Robust-only TSA ----------------
def rsa_matrix(subj: int) -> np.ndarray:
    vecs = []
    for roi in ROI_ORDER:
        D = load_rdm(subj, roi)
        vecs.append(vec_upper(D))
    n = len(ROI_ORDER)
    M = np.full((n, n), np.nan, float)
    for i in range(n):
        for j in range(n):
            M[i, j] = 1.0 if i == j else corr_safe(vecs[i], vecs[j])
    return M


def tsa_matrix_robust_only(subj: int, tgt_sub: pd.DataFrame) -> np.ndarray:
    """
    TSA similarity matrix using robust-only reduced PDs.
    """
    # build robust-only diagrams per ROI
    dgm_by_roi = {}
    for roi in ROI_ORDER:
        raw = load_dgm_h1(subj, roi)
        subroi = tgt_sub[tgt_sub["roi"] == roi]
        targets = subroi[["birth_target", "death_target"]].values.astype(float) if len(subroi) else np.empty((0, 2))
        dgm_by_roi[roi] = build_robust_only_pd(raw, targets)

    n = len(ROI_ORDER)
    Dmat = np.zeros((n, n), float)
    for i, roi_i in enumerate(ROI_ORDER):
        for j, roi_j in enumerate(ROI_ORDER):
            if i == j:
                Dmat[i, j] = 0.0
            elif j < i:
                Dmat[i, j] = Dmat[j, i]
            else:
                Dmat[i, j] = pd_distance(dgm_by_roi[roi_i], dgm_by_roi[roi_j], metric=TSA_METRIC)
                Dmat[j, i] = Dmat[i, j]

    return dist_to_sim(Dmat)


# ---------------- Plotting ----------------
def plot_heatmap(ax, M, title, vmin=None, vmax=None):
    im = ax.imshow(M, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(range(len(ROI_ORDER)))
    ax.set_yticks(range(len(ROI_ORDER)))
    ax.set_xticklabels(ROI_ORDER, rotation=45, ha="right")
    ax.set_yticklabels(ROI_ORDER)
    return im


def plot_side_by_side(rsa, tsa, title, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = plot_heatmap(axes[0], rsa, "RSA similarity (RDM corr)", vmin=-1, vmax=1)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = plot_heatmap(axes[1], tsa, f"Robust-only TSA (H1 targets ≥ {HITRATE_MIN:.2f})", vmin=0, vmax=1)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_scatter(rsa, tsa, title, outpath):
    iu = np.triu_indices(len(ROI_ORDER), k=1)
    x = rsa[iu]
    y = tsa[iu]
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel("RSA similarity (RDM corr)")
    plt.ylabel("Robust-only TSA similarity")
    r = corr_safe(x, y) if len(x) else np.nan
    plt.title(f"{title}\nOff-diagonal corr = {r:.3f}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# ---------------- MAIN ----------------
def main():
    if not TARGETS_CSV.exists():
        raise FileNotFoundError(f"Missing: {TARGETS_CSV}")

    tgt = pd.read_csv(TARGETS_CSV)
    tgt.columns = [c.strip() for c in tgt.columns]

    required = {"subject", "roi", "cluster_id", "hit_rate", "birth_target", "death_target"}
    missing = required - set(tgt.columns)
    if missing:
        raise RuntimeError(f"robust_targets_for_R_repcycles.csv missing columns: {missing}")

    tgt["subject"] = tgt["subject"].astype(int)
    tgt["roi"] = tgt["roi"].astype(str)

    # filter robust targets by your new definition
    tgt = tgt[tgt["hit_rate"] >= HITRATE_MIN].copy()

    if not HAVE_PERSIM:
        print("WARNING: persim not found. Using a crude proxy distance (total persistence difference).")
        print("Install persim for proper diagram distances: pip install persim")

    rsa_all = []
    tsa_all = []
    corr_by_subj = {}

    for subj in SUBJECTS:
        tgt_sub = tgt[tgt["subject"] == subj].copy()

        # If a subject has no robust targets at all, TSA will be all-ones (dist=0) or degenerate.
        # We still compute it but warn.
        if len(tgt_sub) == 0:
            print(f"[WARN] subj{subj:02d} has 0 robust targets at hit_rate ≥ {HITRATE_MIN:.2f}")

        print(f"Computing RSA + robust-only TSA for subj{subj:02d} ...")
        rsa = rsa_matrix(subj)
        tsa = tsa_matrix_robust_only(subj, tgt_sub)

        rsa_all.append(rsa)
        tsa_all.append(tsa)

        # agreement corr
        iu = np.triu_indices(len(ROI_ORDER), k=1)
        r = corr_safe(rsa[iu], tsa[iu])
        corr_by_subj[subj] = r
        print(f"  off-diagonal corr(RSA, robust-only TSA) = {r:.3f}")

        # visuals
        plot_side_by_side(
            rsa, tsa,
            f"subj{subj:02d} — RSA vs robust-only TSA (hit_rate ≥ {HITRATE_MIN:.2f})",
            OUT_DIR / f"subj{subj:02d}_RSA_vs_TSA_robustOnly.png"
        )
        plot_scatter(
            rsa, tsa,
            f"subj{subj:02d}",
            OUT_DIR / f"subj{subj:02d}_RSA_vs_TSA_scatter_robustOnly.png"
        )

    # group average
    rsa_mean = np.nanmean(np.stack(rsa_all, axis=0), axis=0)
    tsa_mean = np.nanmean(np.stack(tsa_all, axis=0), axis=0)

    plot_side_by_side(
        rsa_mean, tsa_mean,
        f"Group average — RSA vs robust-only TSA (hit_rate ≥ {HITRATE_MIN:.2f})",
        OUT_DIR / "GROUPAVG_RSA_vs_TSA_robustOnly.png"
    )
    plot_scatter(
        rsa_mean, tsa_mean,
        "Group average",
        OUT_DIR / "GROUPAVG_RSA_vs_TSA_scatter_robustOnly.png"
    )

    # save correlation summary
    dfc = pd.DataFrame({"subject": list(corr_by_subj.keys()),
                        "rsa_tsa_corr": list(corr_by_subj.values())}).sort_values("subject")
    dfc.to_csv(OUT_DIR / "corr_by_subject.csv", index=False)

    print("\nSaved outputs to:", OUT_DIR)
    print("Per-subject RSA vs TSA correlations:")
    for s in SUBJECTS:
        print(f"  subj{s:02d}: {corr_by_subj.get(s, np.nan):.3f}")


if __name__ == "__main__":
    main()