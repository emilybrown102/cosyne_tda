#!/usr/bin/env python
"""
bootstrap_strict_robust_H1_export_targets.py

What this does:
  1) Bootstraps PH (H1) on subsampled RDMs
  2) Clusters H1 features in (birth,death) space
  3) Defines ROBUST like your old script: hit_rate >= ROBUST_HITRATE_MIN (default 0.80)
  4) Writes:
      - bootstrap_topL_features_H1.csv (per subj×ROI)
      - consensus_clusters_H1.csv      (per subj×ROI)
      - robust_clusters_H1.csv         (master across all subj×ROI)
      - robust_targets_for_R_repcycles.csv  (only robust loops, with birth/death)

This script does NOT try to "perfectly" draw VR graphs.
Instead it exports the robust targets so R/TDA can compute PH-returned representative cycles cleanly.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from ripser import ripser

# ---------------- CONFIG ----------------
ROOT = Path(r"D:\Emily\NSD_cogsci_sandbox")
RDM_ROOT = ROOT / "COGS401" / "COSYNE" / "RDMs_Final" / "subjects"

STRICT_TAG = "STRICT2"  # "STRICT2" or "STRICT3"
SUBJECTS = [1, 2, 3, 4, 5]
ROIS = ["V1", "V2", "V3", "V4", "LO2", "MT", "PH", "STSva", "PIT"]

B = 200
SUBSAMPLE_FRAC = 0.80
RANDOM_SEED = 123

MAXDIM = 1
SHIFT_EPS = 1e-9
PERSIST_FLOOR_FRAC = 0.05   # keep as-is initially (old scripts often did this)
TOP_L = 10

# --- Match old-script behavior ---
ROBUST_HITRATE_MIN = 0.80    # THIS is the big difference vs your new permissive version
CLUSTER_TOL_FRAC = 0.012     # closer to your old value (tighter than 0.02)

OUT_ROOT = (
    ROOT / "COGS401" / "COSYNE" / "RDMs_Final" / "PH_VR_ripser"
    / f"bootstrap{B}_subsample_consensus_{STRICT_TAG}_HIT{int(ROBUST_HITRATE_MIN*100)}_tol{CLUSTER_TOL_FRAC}"
)
OUT_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------- Helpers ----------------
def load_rdm(subj: int, roi: str) -> np.ndarray:
    f = RDM_ROOT / f"subj{subj:02d}" / f"ROI-{roi}_crossnobis_{STRICT_TAG}_sessionSplit_raw.npy"
    if not f.exists():
        raise FileNotFoundError(f"Missing RDM: {f}")
    D = np.load(f).astype(np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"RDM not square: {f} shape={D.shape}")
    if not np.isfinite(D).all():
        raise ValueError(f"RDM has non-finite values: {f}")
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D

def shift_to_nonnegative(D: np.ndarray, eps: float = 1e-9) -> Tuple[np.ndarray, float]:
    n = D.shape[0]
    off = D[~np.eye(n, dtype=bool)]
    m = float(np.min(off))
    if m >= 0:
        return D.copy(), 0.0
    shift = (-m) + eps
    D2 = D + shift
    np.fill_diagonal(D2, 0.0)
    return D2, shift

def max_offdiag(D: np.ndarray) -> float:
    n = D.shape[0]
    off = D[~np.eye(n, dtype=bool)]
    return float(np.max(off))

def subsample_indices(n: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    k = int(round(frac * n))
    k = max(5, min(n, k))
    return rng.choice(n, size=k, replace=False)

def ripser_H1(D: np.ndarray):
    return ripser(D, distance_matrix=True, maxdim=MAXDIM, do_cocycles=False)

def diagram_H1(res) -> np.ndarray:
    dgms = res["dgms"]
    return dgms[1] if len(dgms) > 1 else np.zeros((0, 2), dtype=float)

def finite_persistence(dgm: np.ndarray) -> np.ndarray:
    if dgm.size == 0:
        return np.zeros((0,), dtype=float)
    b = dgm[:, 0]
    d = dgm[:, 1]
    finite = np.isfinite(d)
    pers = np.zeros_like(b)
    pers[finite] = d[finite] - b[finite]
    pers[~finite] = np.nan
    return pers

@dataclass
class Feature:
    birth: float
    death: float
    persistence: float
    maxdist: float
    boot_i: int

def extract_top_features(dgmH1: np.ndarray, maxdist_here: float, top_L: int, floor_frac: float, boot_i: int) -> List[Feature]:
    if dgmH1.size == 0:
        return []
    pers = finite_persistence(dgmH1)
    finite = np.isfinite(pers)
    if not finite.any():
        return []
    births = dgmH1[finite, 0]
    deaths = dgmH1[finite, 1]
    pers_f = pers[finite]

    floor = floor_frac * maxdist_here
    keep = pers_f >= floor
    births, deaths, pers_f = births[keep], deaths[keep], pers_f[keep]
    if births.size == 0:
        return []

    order = np.argsort(-pers_f)[: min(top_L, len(pers_f))]
    return [Feature(float(births[j]), float(deaths[j]), float(pers_f[j]), float(maxdist_here), int(boot_i)) for j in order]

def cluster_features(features: List[Feature], tol: float) -> List[List[Feature]]:
    clusters: List[List[Feature]] = []
    for f in features:
        placed = False
        for c in clusters:
            cb = float(np.mean([x.birth for x in c]))
            cd = float(np.mean([x.death for x in c]))
            if max(abs(f.birth - cb), abs(f.death - cd)) <= tol:
                c.append(f)
                placed = True
                break
        if not placed:
            clusters.append([f])
    return clusters

def centroid(cluster: List[Feature]) -> Tuple[float, float, float]:
    b = float(np.mean([x.birth for x in cluster]))
    d = float(np.mean([x.death for x in cluster]))
    p = float(np.mean([x.persistence for x in cluster]))
    return b, d, p

def run_bootstrap(D_full: np.ndarray, rng: np.random.Generator) -> Tuple[pd.DataFrame, List[List[Feature]]]:
    D_shift, _ = shift_to_nonnegative(D_full, eps=SHIFT_EPS)
    n = D_shift.shape[0]
    all_feats: List[Feature] = []

    for b in range(B):
        idx = subsample_indices(n, SUBSAMPLE_FRAC, rng)
        Ds = D_shift[np.ix_(idx, idx)]
        md = max_offdiag(Ds)
        res = ripser_H1(Ds)
        dgm = diagram_H1(res)
        all_feats.extend(extract_top_features(dgm, md, TOP_L, PERSIST_FLOOR_FRAC, boot_i=b))

    if not all_feats:
        return pd.DataFrame(), []

    md_full = max_offdiag(D_shift)
    tol = CLUSTER_TOL_FRAC * md_full
    clusters = cluster_features(all_feats, tol=tol)

    feats_df = pd.DataFrame([{
        "boot": f.boot_i,
        "birth": f.birth,
        "death": f.death,
        "persistence": f.persistence,
        "maxdist_boot": f.maxdist,
    } for f in all_feats])

    return feats_df, clusters

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    robust_master_rows = []
    targets_for_R = []

    for subj in SUBJECTS:
        for roi in ROIS:
            print(f"\n=== subj{subj:02d} {roi} ===")
            D = load_rdm(subj, roi)
            D_shift, shift_added = shift_to_nonnegative(D, eps=SHIFT_EPS)
            md_full = max_offdiag(D_shift)

            feats_df, clusters = run_bootstrap(D, rng)
            if feats_df.empty or not clusters:
                print("  No H1 features survived floor/top-L in bootstraps.")
                continue

            summaries = []
            for ci, cl in enumerate(clusters):
                hits = len(set([f.boot_i for f in cl]))
                hit_rate = hits / B
                b0, d0, p0 = centroid(cl)
                summaries.append({
                    "cluster_id": ci,
                    "hits": hits,
                    "hit_rate": hit_rate,
                    "birth": b0,
                    "death": d0,
                    "persistence": p0,
                })

            summ_df = pd.DataFrame(summaries).sort_values(["hit_rate", "persistence"], ascending=[False, False])
            summ_df["robust"] = summ_df["hit_rate"] >= ROBUST_HITRATE_MIN

            out_dir = OUT_ROOT / f"subj{subj:02d}" / f"ROI-{roi}"
            out_dir.mkdir(parents=True, exist_ok=True)
            feats_df.to_csv(out_dir / "bootstrap_topL_features_H1.csv", index=False)
            summ_df.to_csv(out_dir / "consensus_clusters_H1.csv", index=False)

            # Master rows + R targets
            for _, r in summ_df.iterrows():
                row = {
                    "subject": subj,
                    "roi": roi,
                    "cluster_id": int(r["cluster_id"]),
                    "hits": int(r["hits"]),
                    "hit_rate": float(r["hit_rate"]),
                    "birth": float(r["birth"]),
                    "death": float(r["death"]),
                    "persistence": float(r["persistence"]),
                    "robust": bool(r["robust"]),
                    "B": int(B),
                    "subsample_frac": float(SUBSAMPLE_FRAC),
                    "persist_floor_frac": float(PERSIST_FLOOR_FRAC),
                    "top_L": int(TOP_L),
                    "cluster_tol_frac": float(CLUSTER_TOL_FRAC),
                    "robust_hitrate_min": float(ROBUST_HITRATE_MIN),
                    "strict_tag": STRICT_TAG,
                    "maxdist_full": float(md_full),
                    "shift_added": float(shift_added),
                }
                robust_master_rows.append(row)

                if row["robust"]:
                    targets_for_R.append({
                        "subject": subj,
                        "roi": roi,
                        "cluster_id": row["cluster_id"],
                        "hit_rate": row["hit_rate"],
                        "birth_target": row["birth"],
                        "death_target": row["death"],
                        "strict_tag": STRICT_TAG,
                    })

            n_rob = int(summ_df["robust"].sum())
            print(f"  Robust clusters (hit_rate >= {ROBUST_HITRATE_MIN:.2f}): {n_rob}")

    # Write master files
    master_csv = OUT_ROOT / "robust_clusters_H1.csv"
    pd.DataFrame(robust_master_rows).to_csv(master_csv, index=False)

    targets_csv = OUT_ROOT / "robust_targets_for_R_repcycles.csv"
    pd.DataFrame(targets_for_R).to_csv(targets_csv, index=False)

    print(f"\n✅ Wrote: {master_csv}")
    print(f"✅ Wrote: {targets_csv}")

if __name__ == "__main__":
    main()