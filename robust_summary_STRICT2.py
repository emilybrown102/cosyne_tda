#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- CONFIG ----------------
ROOT = Path(r"D:\Emily\NSD_cogsci_sandbox\COGS401\COSYNE\RDMs_Final")

# Use TARGETS (true robust loops) for all counts/plots
TARGETS_CSV = ROOT / "PH_VR_ripser" / "bootstrap_results" / "robust_targets_for_R_repcycles.csv"
# Optional: load clusters file only for a sanity print
CLUSTERS_CSV = ROOT / "PH_VR_ripser" / "bootstrap_results" / "robust_clusters_H1.csv"

# Save outputs here (as requested)
OUT_DIR = ROOT / "poster_panels" / "robust_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- LOAD ----------------
if not TARGETS_CSV.exists():
    raise FileNotFoundError(f"Could not find: {TARGETS_CSV}")

tgt = pd.read_csv(TARGETS_CSV)
tgt.columns = [c.strip() for c in tgt.columns]

required = {"subject", "roi", "cluster_id", "hit_rate"}
missing = required - set(tgt.columns)
if missing:
    raise RuntimeError(
        f"{TARGETS_CSV.name} is missing columns: {missing}\n"
        f"Found columns: {list(tgt.columns)}"
    )

# Make subject sortable
tgt["subject"] = tgt["subject"].astype(int)

# Unique robust-loop key (cluster_id can repeat across subj/ROI)
tgt["cluster_key"] = (
    tgt["subject"].astype(str) + "|" +
    tgt["roi"].astype(str) + "|" +
    tgt["cluster_id"].astype(str)
)

# ---------------- SANITY PRINTS ----------------
print("Loaded targets:", TARGETS_CSV)
print("Rows in targets (≈ # robust loops):", len(tgt))
print("Unique robust loops (subj×ROI×cluster_id):", tgt["cluster_key"].nunique())

# Optional: show why clusters file can look huge
if CLUSTERS_CSV.exists():
    cl = pd.read_csv(CLUSTERS_CSV)
    cl.columns = [c.strip() for c in cl.columns]
    print("\n(For reference) clusters file:", CLUSTERS_CSV)
    print("Rows in robust_clusters_H1.csv:", len(cl))
    if {"subject", "roi", "cluster_id"}.issubset(cl.columns):
        try:
            cl["subject"] = cl["subject"].astype(int)
        except Exception:
            pass
        cl["cluster_key"] = (
            cl["subject"].astype(str) + "|" + cl["roi"].astype(str) + "|" + cl["cluster_id"].astype(str)
        )
        print("Unique subj×ROI×cluster_id in clusters file:", cl["cluster_key"].nunique())

# ---------------- PLOT 1: # robust loops per ROI ----------------
cnt_roi = tgt.groupby("roi")["cluster_key"].nunique().sort_values(ascending=False)

plt.figure(figsize=(10, 4))
cnt_roi.plot(kind="bar")
plt.ylabel("# robust H1 loops")
plt.title("Robust H1 loops per ROI (STRICT2)")
plt.tight_layout()
plt.savefig(OUT_DIR / "robust_count_per_roi.png", dpi=300)
plt.close()

# ---------------- PLOT 2: max hit_rate per ROI ----------------
mx_hit = tgt.groupby("roi")["hit_rate"].max().sort_values(ascending=False)

plt.figure(figsize=(10, 4))
mx_hit.plot(kind="bar")
plt.ylim(0, 1.0)
plt.ylabel("max hit_rate")
plt.title("Max robustness (hit_rate) per ROI (STRICT2)")
plt.tight_layout()
plt.savefig(OUT_DIR / "robust_max_hitrate_per_roi.png", dpi=300)
plt.close()

# ---------------- PLOT 3: per-subject counts (heatmap-style table) ----------------
tab = (tgt.groupby(["subject", "roi"])["cluster_key"]
         .nunique()
         .unstack(fill_value=0)
         .sort_index())

plt.figure(figsize=(12, 4))
plt.imshow(tab.values, aspect="auto")
plt.xticks(range(tab.shape[1]), tab.columns, rotation=45, ha="right")
plt.yticks(range(tab.shape[0]), tab.index)
plt.colorbar(label="# robust loops")
plt.title("Robust H1 loops per subject × ROI (STRICT2)")
plt.tight_layout()
plt.savefig(OUT_DIR / "robust_counts_subject_by_roi_heatmap.png", dpi=300)
plt.close()

# ---------------- PLOT 4: hit_rate distribution ----------------
plt.figure(figsize=(6, 4))
plt.hist(tgt["hit_rate"].values, bins=15)
plt.xlabel("hit_rate")
plt.ylabel("count")
plt.title("Distribution of robust-loop hit rates (STRICT2)")
plt.tight_layout()
plt.savefig(OUT_DIR / "robust_hit_rate_hist.png", dpi=300)
plt.close()

print("\nSaved robustness summary outputs to:", OUT_DIR)
print("Total robust loops (unique subj×ROI×cluster_id):", tgt["cluster_key"].nunique())
print("ROIs:", ", ".join(sorted(tgt["roi"].unique())))