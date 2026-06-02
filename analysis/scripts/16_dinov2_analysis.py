"""
16_dinov2_analysis.py
======================
Validation of CP3D phenotypic clusters in a deep-learning-derived feature space.

Mirrors the cluster analysis of script 11 but operates on the 1536-dimensional
DINOv2 embeddings (4 Cell Painting channels x 384-dim ViT-S/14 CLS tokens) that
were generated from the raw MIP images by script 15. This provides an
independent, learned-from-pixels representation orthogonal to the
hand-engineered CellProfiler feature set used by the parent CP3D pipeline.

The script answers three questions:

1. Does the HSP90 ansamycin cluster (geldanamycin, retaspimycin, alvespimycin)
   reach statistical significance in the DINOv2 phenotypic space?
2. Do the inter-compound distances among the 25 robust 3D hits agree between
   CellProfiler and DINOv2 spaces (Mantel test)?
3. Which Cell Painting channel (ER, AGP, MITO, DNA) contributes most to the
   HSP90 cluster signal?

Inputs
------
data/embeddings/embeddings_per_compound.parquet     (script 15 output)
data/annotated/cp3d_library_annotated.csv           (script 10 output)
data/processed/consensus_C2386.csv, consensus_C2387.csv, consensus_C2388.csv
    (used to compute CellProfiler distance matrix for the Mantel test)

Outputs (saved in results/dinov2_analysis/)
-------------------------------------------
merged_compound_embeddings.csv    Library annotation + DINOv2 features merged
hsp90_cluster_dinov2.csv          Permutation test results
mantel_test_results.csv           CP vs DINOv2 distance matrix correlation
per_channel_hsp90_test.csv        Per-channel cluster strength
summary.txt                       Narrative summary
figures/
    umap_dinov2_space.png         UMAP of 735 compounds, hits and HSP90 highlighted
    hsp90_permutation_dinov2.png  Null distribution + observed cluster distance
    distance_correlation.png      CP vs DINOv2 distance matrix scatter (25 hits)
    per_channel_strength.png      Channel-wise contribution to HSP90 cluster

Usage
-----
    python 16_dinov2_analysis.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import pdist, squareform

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(r'C:\Users\Ianezi\Documents\CP3D')
ANALYSIS = BASE / 'analysis'
EMB_FILE = ANALYSIS / 'data' / 'embeddings' / 'embeddings_per_compound.parquet'
LIB_FILE = ANALYSIS / 'data' / 'annotated' / 'cp3d_library_annotated.csv'
PROC = ANALYSIS / 'data' / 'processed'
OUT = ANALYSIS / 'results' / 'dinov2_analysis'
FIG = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CONSENSUS_3D = {
    'C2386': PROC / 'consensus_C2386.csv',
    'C2387': PROC / 'consensus_C2387.csv',
    'C2388': PROC / 'consensus_C2388.csv',
}

CHANNEL_NAMES = ['ER', 'AGP', 'MITO', 'DNA']
HSP90_ANSAMYCINS = ['GELDANAMYCIN', 'RETASPIMYCIN', 'ALVESPIMYCIN']
N_PERMUTATIONS = 10000
RANDOM_SEED = 42

print(f"\n{'='*70}\n=== DINOv2 phenotypic analysis ===\n{'='*70}\n")

# ---------------------------------------------------------------------------
# 1. Load + merge
# ---------------------------------------------------------------------------
print(f"{'-'*70}\n1. Loading embeddings and library\n{'-'*70}")
emb = pd.read_parquet(EMB_FILE)
lib = pd.read_csv(LIB_FILE)
print(f"  DINOv2 embeddings: {emb.shape}")
print(f"  Library:           {lib.shape}, hits: {int(lib['is_hit_3D'].sum())}")

merged = lib.merge(emb, on='EOS_id', how='inner')
print(f"  Merged (inner):    {merged.shape}, hits: {int(merged['is_hit_3D'].sum())}")
merged.to_csv(OUT / 'merged_compound_embeddings.csv', index=False)

feature_cols = [c for c in emb.columns if c != 'EOS_id']
n_features = len(feature_cols)
print(f"  Feature dimensions: {n_features} (= 4 channels x {n_features // 4})")

# ---------------------------------------------------------------------------
# 2. Standardize features (z-score within the analytical library)
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n2. Standardizing features (z-score within library)\n{'-'*70}")
X = merged[feature_cols].values.astype(float)
X = np.nan_to_num(X, nan=0.0)
mu = X.mean(axis=0)
sd = X.std(axis=0) + 1e-9
Xz = (X - mu) / sd
print(f"  Standardized matrix: {Xz.shape}")

# Distance matrix in full DINOv2 space
D_full = squareform(pdist(Xz, metric='euclidean'))
eos_index = merged['EOS_id'].values
hits_idx = np.where(merged['is_hit_3D'].values)[0]
hsp90_eos = merged.loc[
    merged['Drug_name'].fillna('').str.upper().isin(HSP90_ANSAMYCINS),
    'EOS_id'].tolist()
hsp90_idx = [i for i, e in enumerate(eos_index) if e in set(hsp90_eos)]
print(f"\n  Pool sizes:")
print(f"    Library:    {len(eos_index)}")
print(f"    3D hits:    {len(hits_idx)}")
print(f"    HSP90:      {len(hsp90_idx)} ({hsp90_eos})")

# ---------------------------------------------------------------------------
# 3. HSP90 cluster permutation test in DINOv2 space
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n3. HSP90 cluster permutation test (DINOv2 space)\n{'-'*70}")
rng = np.random.default_rng(RANDOM_SEED)

def cluster_p_value(D, target_idx, pool_idx, n_perm, rng):
    """Permutation p-value: probability that a random k-tuple from `pool_idx`
    has mean pairwise distance <= mean of target group."""
    k = len(target_idx)
    pairs = [(i, j) for i in target_idx for j in target_idx if j > i]
    mean_target = float(np.mean([D[i, j] for i, j in pairs]))
    nulls = []
    for _ in range(n_perm):
        idx = rng.choice(pool_idx, size=k, replace=False)
        pp = [(idx[i], idx[j]) for i in range(k) for j in range(i+1, k)]
        nulls.append(np.mean([D[a, b] for a, b in pp]))
    nulls = np.array(nulls)
    p = float((nulls <= mean_target).sum() / n_perm)
    return mean_target, float(nulls.mean()), float(nulls.std()), p, nulls

if len(hsp90_idx) >= 2 and len(hits_idx) >= 4:
    # vs hit triplets
    mt_h, nm_h, ns_h, p_h, null_h = cluster_p_value(
        D_full, hsp90_idx, hits_idx.tolist(), N_PERMUTATIONS, rng)
    # vs library triplets
    mt_l, nm_l, ns_l, p_l, null_l = cluster_p_value(
        D_full, hsp90_idx, list(range(len(eos_index))), N_PERMUTATIONS, rng)

    print(f"\n  Mean pairwise distance HSP90 in DINOv2 space: {mt_h:.3f}")
    print(f"  Null vs hit triplets (n = {len(hits_idx)}):  null = {nm_h:.3f}, p = {p_h:.4f}")
    print(f"  Null vs library triplets (n = {len(eos_index)}): null = {nm_l:.3f}, p = {p_l:.4f}")

    hsp90_result = pd.DataFrame([{
        'context': 'DINOv2 full (1536-dim)',
        'k_target': len(hsp90_idx),
        'mean_pairwise_distance_target': mt_h,
        'null_mean_within_hits': nm_h,
        'null_sd_within_hits': ns_h,
        'p_within_hits': p_h,
        'null_mean_library': nm_l,
        'null_sd_library': ns_l,
        'p_library': p_l,
        'n_permutations': N_PERMUTATIONS,
    }])
    hsp90_result.to_csv(OUT / 'hsp90_cluster_dinov2.csv', index=False)

    # Plot permutation
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, null, label, pval in [
            (axes[0], null_h, f'Null: random triplets among 3D hits (n = {len(hits_idx)})', p_h),
            (axes[1], null_l, f'Null: random triplets across library (n = {len(eos_index)})', p_l)]:
        ax.hist(null, bins=60, color='#888', alpha=0.7)
        ax.axvline(mt_h if 'hits' in label else mt_l, color='#D7263D', lw=2.5,
                   label=f'HSP90 ansamycins: {mt_h:.2f}')
        ax.set_xlabel('Mean pairwise Euclidean distance (DINOv2 1536-dim)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{label}\nPermutation p = {pval:.4f}')
        ax.legend()
    plt.suptitle('HSP90 ansamycin cluster significance in DINOv2 phenotypic space',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG / 'hsp90_permutation_dinov2.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Figure: {FIG / 'hsp90_permutation_dinov2.png'}")
else:
    print(f"  Insufficient HSP90 targets in embedding ({len(hsp90_idx)}); skipping.")
    hsp90_result = pd.DataFrame()

# ---------------------------------------------------------------------------
# 4. Mantel test: distance matrix correlation between CellProfiler and DINOv2
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n4. Mantel test (CellProfiler vs DINOv2)\n{'-'*70}")

# Load CellProfiler consensus features for the same compounds
consensus_3d_dfs = []
for plate, path in CONSENSUS_3D.items():
    if path.exists():
        d = pd.read_csv(path)
        d = d[d['Metadata_Well_type'] == 'compound'].copy()
        d = d.rename(columns={'Metadata_Compound': 'EOS_id'})
        consensus_3d_dfs.append(d)
c3d = pd.concat(consensus_3d_dfs, ignore_index=True)
feat3d = [c for c in c3d.columns if not c.startswith('Metadata_')
          and c != 'EOS_id' and pd.api.types.is_numeric_dtype(c3d[c])]
# Keep one row per compound (max activity)
def rms(row, cols):
    v = np.asarray(row[cols].values, dtype=float)
    v = v[~np.isnan(v)]
    return float(np.sqrt(np.mean(v**2))) if len(v) else np.nan
c3d['_act'] = c3d.apply(lambda r: rms(r, feat3d), axis=1)
c3d_uniq = c3d.sort_values('_act', ascending=False).drop_duplicates('EOS_id', keep='first')
print(f"  CellProfiler consensus loaded: {len(c3d_uniq)} compounds x {len(feat3d)} features")

# Intersect with DINOv2-merged compounds, keep only hits for Mantel (25)
hits_eos_list = merged.loc[merged['is_hit_3D'], 'EOS_id'].tolist()
cp_hits = c3d_uniq[c3d_uniq['EOS_id'].isin(hits_eos_list)].set_index('EOS_id')
dino_hits = merged[merged['is_hit_3D']][['EOS_id'] + feature_cols].set_index('EOS_id')
common_eos = sorted(set(cp_hits.index) & set(dino_hits.index))
cp_mat = cp_hits.loc[common_eos, feat3d].fillna(0).values
dn_mat = dino_hits.loc[common_eos, feature_cols].fillna(0).values

# Z-score within hit subset
def zscale(m):
    mu_, sd_ = m.mean(axis=0), m.std(axis=0) + 1e-9
    return (m - mu_) / sd_
cp_mat_z = zscale(cp_mat)
dn_mat_z = zscale(dn_mat)

D_cp = squareform(pdist(cp_mat_z, metric='euclidean'))
D_dn = squareform(pdist(dn_mat_z, metric='euclidean'))

# Mantel: correlate the upper-triangular flattened distance matrices
iu = np.triu_indices_from(D_cp, k=1)
v_cp = D_cp[iu]
v_dn = D_dn[iu]
spearman_rho, spearman_p = stats.spearmanr(v_cp, v_dn)
pearson_r, pearson_p = stats.pearsonr(v_cp, v_dn)
# Mantel permutation p (label shuffle)
mantel_nulls = []
for _ in range(N_PERMUTATIONS):
    perm = rng.permutation(len(common_eos))
    D_dn_perm = D_dn[perm][:, perm]
    v_perm = D_dn_perm[iu]
    rho_perm, _ = stats.spearmanr(v_cp, v_perm)
    mantel_nulls.append(rho_perm)
mantel_nulls = np.array(mantel_nulls)
mantel_p = float((mantel_nulls >= spearman_rho).sum() / N_PERMUTATIONS)

print(f"\n  Pairs compared: {len(v_cp)} ({len(common_eos)} hits)")
print(f"  Spearman rho:   {spearman_rho:.3f}  (analytic p = {spearman_p:.3g})")
print(f"  Pearson r:      {pearson_r:.3f}   (analytic p = {pearson_p:.3g})")
print(f"  Mantel permutation p (one-sided greater): {mantel_p:.4f}")

mantel_df = pd.DataFrame([{
    'n_compounds': len(common_eos),
    'n_pairs': len(v_cp),
    'spearman_rho': spearman_rho,
    'spearman_p_analytic': spearman_p,
    'pearson_r': pearson_r,
    'pearson_p_analytic': pearson_p,
    'mantel_p_permutation': mantel_p,
    'n_permutations': N_PERMUTATIONS,
}])
mantel_df.to_csv(OUT / 'mantel_test_results.csv', index=False)

# Plot
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(v_cp, v_dn, s=20, alpha=0.6, color='#3D5A80', edgecolor='black', linewidth=0.3)
# Diagonal reference
mn = min(v_cp.min(), v_dn.min())
mx = max(v_cp.max(), v_dn.max())
ax.plot([mn, mx], [mn, mx], '--', color='#888', lw=0.7, alpha=0.6)
ax.set_xlabel('Pairwise distance, CellProfiler 3D space')
ax.set_ylabel('Pairwise distance, DINOv2 1536-dim space')
ax.set_title(f'Inter-hit distance correlation across feature spaces\n'
             f'Spearman rho = {spearman_rho:.3f}, Mantel p = {mantel_p:.4f} '
             f'(n = {len(common_eos)} hits)')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / 'distance_correlation.png', dpi=200, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------------------------
# 5. Per-channel HSP90 cluster strength
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n5. Per-channel HSP90 cluster strength\n{'-'*70}")
per_channel_rows = []
for ch in CHANNEL_NAMES:
    ch_cols = [c for c in feature_cols if c.startswith(f'{ch}_emb_')]
    if not ch_cols:
        continue
    X_ch = merged[ch_cols].values.astype(float)
    X_ch = np.nan_to_num(X_ch, nan=0.0)
    mu_ch, sd_ch = X_ch.mean(axis=0), X_ch.std(axis=0) + 1e-9
    X_chz = (X_ch - mu_ch) / sd_ch
    D_ch = squareform(pdist(X_chz, metric='euclidean'))
    if len(hsp90_idx) >= 2 and len(hits_idx) >= 4:
        mt, nm, ns, pv, _ = cluster_p_value(
            D_ch, hsp90_idx, hits_idx.tolist(), N_PERMUTATIONS, rng)
        per_channel_rows.append({
            'channel': ch,
            'n_features': len(ch_cols),
            'mean_pairwise_distance_HSP90': mt,
            'null_mean': nm, 'null_sd': ns,
            'p_within_hits': pv,
        })
        print(f"  {ch:6s}: dist HSP90 = {mt:.3f}, null = {nm:.3f}, p = {pv:.4f}")
per_channel_df = pd.DataFrame(per_channel_rows)
per_channel_df.to_csv(OUT / 'per_channel_hsp90_test.csv', index=False)

# Plot per-channel
if len(per_channel_rows):
    fig, ax = plt.subplots(figsize=(8, 5))
    ch_labels = per_channel_df['channel'].values
    p_vals = per_channel_df['p_within_hits'].values
    minus_log_p = -np.log10(np.clip(p_vals, 1e-5, 1))
    colors = ['#D7263D' if p < 0.05 else '#888' for p in p_vals]
    bars = ax.bar(ch_labels, minus_log_p, color=colors, edgecolor='black')
    for bar, p in zip(bars, p_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'p = {p:.3f}', ha='center', fontsize=10)
    ax.axhline(-np.log10(0.05), ls='--', color='#888', lw=0.7,
                label='p = 0.05')
    ax.set_ylabel('$-\\log_{10}$(permutation p)')
    ax.set_xlabel('Cell Painting channel')
    ax.set_title('HSP90 ansamycin cluster strength by channel\n'
                  '(within-hits permutation test in DINOv2 384-dim sub-space)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / 'per_channel_strength.png', dpi=200, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------
# 6. UMAP visualization
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n6. UMAP visualization\n{'-'*70}")
try:
    import umap
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=RANDOM_SEED)
    coords = reducer.fit_transform(Xz)
    merged['UMAP_1'] = coords[:, 0]
    merged['UMAP_2'] = coords[:, 1]
    fig, ax = plt.subplots(figsize=(9, 7))
    # Non-hits
    mask_nh = ~merged['is_hit_3D']
    ax.scatter(merged.loc[mask_nh, 'UMAP_1'], merged.loc[mask_nh, 'UMAP_2'],
                s=10, alpha=0.3, color='#BBBBBB', label=f'Non-hit (n={mask_nh.sum()})')
    # Hits not HSP90
    mask_h = merged['is_hit_3D'] & ~merged['EOS_id'].isin(hsp90_eos)
    ax.scatter(merged.loc[mask_h, 'UMAP_1'], merged.loc[mask_h, 'UMAP_2'],
                s=50, alpha=0.9, color='#377EB8', edgecolor='black', linewidth=0.4,
                label=f'3D hit (n={mask_h.sum()})')
    # HSP90 ansamycins
    mask_hsp = merged['EOS_id'].isin(hsp90_eos)
    ax.scatter(merged.loc[mask_hsp, 'UMAP_1'], merged.loc[mask_hsp, 'UMAP_2'],
                s=120, alpha=0.95, color='#D7263D', edgecolor='black', linewidth=0.6,
                label=f'HSP90 ansamycin (n={mask_hsp.sum()})', marker='*')
    # Annotate HSP90 names
    for _, r in merged.loc[mask_hsp].iterrows():
        drug = str(r['Drug_name']) if pd.notna(r['Drug_name']) else r['EOS_id']
        ax.annotate(drug, (r['UMAP_1'], r['UMAP_2']),
                    xytext=(8, 4), textcoords='offset points', fontsize=10,
                    fontweight='bold', color='#D7263D')
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.set_title('DINOv2 1536-dim embedding space (UMAP projection)\n'
                  'HSP90 ansamycin cluster highlighted')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / 'umap_dinov2_space.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  UMAP plot saved.")
except ImportError:
    print(f"  umap-learn not installed; skipping UMAP.")
    print(f"  Run: pip install umap-learn")

# ---------------------------------------------------------------------------
# 7. Narrative summary
# ---------------------------------------------------------------------------
summary = OUT / 'summary.txt'
with open(summary, 'w', encoding='utf-8') as f:
    f.write("DINOv2 deep embedding analysis of CP3D phenotypic space\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Embedding source: DINOv2 ViT-S/14 (Oquab et al. 2023, arXiv 2304.07193)\n")
    f.write(f"  Feature space: 4 Cell Painting channels x 384-dim CLS tokens = "
            f"{n_features}-dim per compound.\n")
    f.write(f"  Compounds analyzed: {len(merged)} (3D hits: {int(merged['is_hit_3D'].sum())})\n\n")

    if len(hsp90_result):
        f.write("HSP90 ansamycin cluster permutation test (DINOv2 full space)\n")
        f.write("-" * 60 + "\n")
        r = hsp90_result.iloc[0]
        f.write(f"  Mean pairwise distance, HSP90 triplet: "
                f"{r['mean_pairwise_distance_target']:.3f}\n")
        f.write(f"  Null vs random hit triplets:           "
                f"mean = {r['null_mean_within_hits']:.3f}, "
                f"p = {r['p_within_hits']:.4f}\n")
        f.write(f"  Null vs random library triplets:       "
                f"mean = {r['null_mean_library']:.3f}, "
                f"p = {r['p_library']:.4f}\n\n")
        f.write("  Reference (parent CP3D pipeline, CellProfiler space):\n")
        f.write("    3D-MEDINA p_within_hits = 0.043 (stage 9 multivariate)\n")
        f.write("    2D-MEDINA p_within_hits = 0.076 (script 11 trend)\n\n")

    f.write("Mantel test (CellProfiler vs DINOv2 distance matrices)\n")
    f.write("-" * 60 + "\n")
    f.write(f"  Hits analyzed:                          {len(common_eos)}\n")
    f.write(f"  Spearman rho (distance pairs):          {spearman_rho:.3f}\n")
    f.write(f"  Pearson r (distance pairs):             {pearson_r:.3f}\n")
    f.write(f"  Mantel permutation p (one-sided):       {mantel_p:.4f}\n")
    f.write(f"  Interpretation: rho > 0 with low p means the two phenotypic\n")
    f.write(f"  spaces (CellProfiler hand-engineered vs DINOv2 self-supervised)\n")
    f.write(f"  agree on how to group the 25 robust 3D hits.\n\n")

    if len(per_channel_rows):
        f.write("Per-channel HSP90 cluster strength (DINOv2 384-dim sub-spaces)\n")
        f.write("-" * 60 + "\n")
        f.write(per_channel_df.to_string(index=False))
        f.write("\n")
print(f"\n  Summary file: {summary}")
print(f"\n{'='*70}\nAnalysis complete. Outputs in: {OUT}\n{'='*70}\n")
