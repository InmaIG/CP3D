"""
compare_hsp90_stk24.py
======================
Side-by-side feature comparison of the HSP90 triplet vs the STK24 triplet.

Inputs
------
    data/processed/consensus_<plate>.csv

Outputs (results/hsp90_vs_stk24/)
--------------------------------
    heatmap_6cpds_x_features.png       6 compounds x N features (clustered)
    correlation_matrix_6x6.png         Pearson correlation between the 6 profiles
    correlation_matrix_6x6.csv         (same as a CSV)
    top_discriminating_features.png    Top features differing HSP90 vs STK24
    multiorganelle_signature.png       Granularity ER+AGP+MITO scales 2-4
    summary.txt                        Numerical takeaways
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# ---------------- Paths ----------------
BASE = Path(r'C:\Users\Ianezi\Documents\CP3D\analysis')
PROC_DIR = BASE / 'data' / 'processed'
OUT_DIR = BASE / 'results' / 'hsp90_vs_stk24'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLATES = ['C2386', 'C2387', 'C2388']

# The 6 compounds
GROUPS = {
    'HSP90': [
        ('EOS101988', 'GELDANAMYCIN', 'C2387'),
        ('EOS100193', 'ALVESPIMYCIN', 'C2387'),
        ('EOS100198', 'RETASPIMYCIN', 'C2387'),
    ],
    'STK24': [
        ('EOS100486', 'G-5555',      'C2387'),
        ('EOS100538', 'DOVITINIB',   'C2387'),
        ('EOS101776', 'LOSMAPIMOD',  'C2386'),
    ],
}
GROUP_COLOR = {'HSP90': '#00807A', 'STK24': '#B0362A'}

# ---------------- Load consensus profiles ----------------
print("STEP 1: Load consensus profiles and intersect features")
per_plate = {}
for plate in PLATES:
    per_plate[plate] = pd.read_csv(PROC_DIR / f'consensus_{plate}.csv')
    print(f"  {plate}: {per_plate[plate].shape}")

# Common features across plates (same logic as script 09)
meta_prefixes = ('Metadata_', 'is_', 'Hit_', 'Phenotype')
feat_sets = {}
for plate, df in per_plate.items():
    feats = [c for c in df.columns if not any(c.startswith(p) for p in meta_prefixes)]
    feat_sets[plate] = set(feats)
common_features = sorted(set.intersection(*feat_sets.values()))
print(f"  Common features: {len(common_features)}")

# ---------------- Build 6-compound matrix ----------------
print("\nSTEP 2: Extract the 6 profiles")
rows = []
labels = []
group_labels = []
plate_labels = []
for group, members in GROUPS.items():
    for eos, drug, plate in members:
        df = per_plate[plate]
        match = df[df['Metadata_Compound'] == eos]
        if match.empty:
            print(f"  WARNING: {drug} ({eos}) not in {plate}")
            continue
        # Take the first (consensus is already aggregated)
        row = match.iloc[0][common_features].values.astype(float)
        rows.append(row)
        labels.append(drug)
        group_labels.append(group)
        plate_labels.append(plate)
        print(f"  {group:6s}  {drug:14s}  {plate}  -> {(~np.isnan(row)).sum()}/{len(common_features)} features")

X = np.array(rows)
X = np.nan_to_num(X, nan=0.0)
print(f"  Matrix: {X.shape}")

# ---------------- Output 1: Heatmap 6 x features ----------------
print("\nSTEP 3: Heatmap 6 x features (Ward clustering on both axes)")

# Cluster features
feat_link = linkage(pdist(X.T), method='ward')
feat_order = leaves_list(feat_link)

# Cluster compounds
cpd_link = linkage(pdist(X), method='ward')
cpd_order = leaves_list(cpd_link)

X_ord = X[cpd_order][:, feat_order]
labels_ord = [labels[i] for i in cpd_order]
group_ord  = [group_labels[i] for i in cpd_order]

fig, ax = plt.subplots(figsize=(14, 4.5))
vmax = np.percentile(np.abs(X_ord), 98)
im = ax.imshow(X_ord, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
ax.set_yticks(range(len(labels_ord)))
yticklabels = [f"{lbl}  ({grp})" for lbl, grp in zip(labels_ord, group_ord)]
ax.set_yticklabels(yticklabels)
for i, grp in enumerate(group_ord):
    ax.get_yticklabels()[i].set_color(GROUP_COLOR[grp])
    ax.get_yticklabels()[i].set_fontweight('bold')
ax.set_xticks([])
ax.set_xlabel(f'{len(common_features)} common features (Ward-clustered)')
ax.set_title('Feature profiles · HSP90 (teal) vs STK24 (coral) — '
             '6 compounds × 112 features (z-score)', fontsize=11)
plt.colorbar(im, ax=ax, label='z-score', shrink=0.7)
plt.tight_layout()
plt.savefig(OUT_DIR / 'heatmap_6cpds_x_features.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"  Saved: heatmap_6cpds_x_features.png")
print(f"  Compound order (top -> bottom): {labels_ord}")

# ---------------- Output 2: Correlation matrix ----------------
print("\nSTEP 4: 6x6 correlation matrix")
n = len(labels)
corr = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i == j:
            corr[i, j] = 1.0
        else:
            r, _ = pearsonr(X[i], X[j])
            corr[i, j] = r

corr_df = pd.DataFrame(corr, index=labels, columns=labels)
corr_df.to_csv(OUT_DIR / 'correlation_matrix_6x6.csv', float_format='%.3f')

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{corr[i, j]:.2f}", ha='center', va='center',
                fontsize=9, color='black' if abs(corr[i, j]) < 0.6 else 'white',
                fontweight='bold')
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels)
# Color tick labels by group
for i, grp in enumerate(group_labels):
    ax.get_xticklabels()[i].set_color(GROUP_COLOR[grp])
    ax.get_xticklabels()[i].set_fontweight('bold')
    ax.get_yticklabels()[i].set_color(GROUP_COLOR[grp])
    ax.get_yticklabels()[i].set_fontweight('bold')
# Frame the two groups
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((-0.5, -0.5), 3, 3, fill=False,
                       edgecolor=GROUP_COLOR['HSP90'], lw=2.5))
ax.add_patch(Rectangle((2.5, 2.5), 3, 3, fill=False,
                       edgecolor=GROUP_COLOR['STK24'], lw=2.5))
plt.colorbar(im, ax=ax, label='Pearson r')
ax.set_title('Pearson correlation between compound profiles\n'
             '(112 common features)', fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / 'correlation_matrix_6x6.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"  Saved: correlation_matrix_6x6.png + .csv")

# ---------------- Output 3: Top discriminating features ----------------
print("\nSTEP 5: Top features discriminating HSP90 vs STK24")
hsp_idx = [i for i, g in enumerate(group_labels) if g == 'HSP90']
stk_idx = [i for i, g in enumerate(group_labels) if g == 'STK24']
hsp_mean = X[hsp_idx].mean(axis=0)
stk_mean = X[stk_idx].mean(axis=0)
diff = hsp_mean - stk_mean

top_n = 15
top_idx = np.argsort(-np.abs(diff))[:top_n]
top_feats = [common_features[i] for i in top_idx]
top_diff = diff[top_idx]

# Shorten feature names for readability
def shorten(name, maxlen=44):
    return name if len(name) <= maxlen else name[:maxlen-3] + '...'

fig, ax = plt.subplots(figsize=(11, 7))
y = np.arange(top_n)
colors = [GROUP_COLOR['HSP90'] if d > 0 else GROUP_COLOR['STK24'] for d in top_diff]
ax.barh(y, top_diff, color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(y)
ax.set_yticklabels([shorten(f) for f in top_feats], fontsize=9)
ax.invert_yaxis()
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('Mean z-score HSP90 − mean z-score STK24')
ax.set_title('Top 15 features that distinguish HSP90 from STK24\n'
             '(green bar = higher in HSP90 · coral = higher in STK24)', fontsize=11)
# Annotate with the two means
for i, idx in enumerate(top_idx):
    ax.text(top_diff[i] + (0.3 if top_diff[i] > 0 else -0.3), i,
            f"  H={hsp_mean[idx]:+.1f}  S={stk_mean[idx]:+.1f}",
            va='center', fontsize=7,
            ha='left' if top_diff[i] > 0 else 'right',
            color='gray')
plt.tight_layout()
plt.savefig(OUT_DIR / 'top_discriminating_features.png', dpi=140, bbox_inches='tight')
plt.close()
print(f"  Saved: top_discriminating_features.png")

# ---------------- Output 4: Multi-organelle signature ----------------
print("\nSTEP 6: Multi-organelle signature (Granularity ER/AGP/MITO scales 2-4)")

# Find the 9 granularity features (3 channels x 3 scales)
signature_feats = []
for ch in ('ER', 'AGP', 'MITO', 'Mito'):
    for scale in (2, 3, 4):
        candidates = [f for f in common_features
                      if 'Granularity' in f and f'_{scale}_' in f and ch in f]
        if candidates:
            signature_feats.append(candidates[0])

# Deduplicate while preserving order
seen = set()
signature_feats = [f for f in signature_feats if not (f in seen or seen.add(f))]
print(f"  Found {len(signature_feats)} signature features")
for f in signature_feats:
    print(f"    {f}")

if signature_feats:
    sig_idx = [common_features.index(f) for f in signature_feats]
    sig_X = X[:, sig_idx]  # 6 x n_sig

    fig, ax = plt.subplots(figsize=(11, 5.5))
    vmax_sig = np.percentile(np.abs(sig_X), 99)
    im = ax.imshow(sig_X, aspect='auto', cmap='RdBu_r', vmin=-vmax_sig, vmax=vmax_sig)
    for i in range(sig_X.shape[0]):
        for j in range(sig_X.shape[1]):
            val = sig_X[i, j]
            ax.text(j, i, f"{val:+.1f}", ha='center', va='center',
                    fontsize=8.5,
                    color='black' if abs(val) < 0.5 * vmax_sig else 'white',
                    fontweight='bold')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f"{l} ({g})" for l, g in zip(labels, group_labels)])
    for i, grp in enumerate(group_labels):
        ax.get_yticklabels()[i].set_color(GROUP_COLOR[grp])
        ax.get_yticklabels()[i].set_fontweight('bold')
    ax.set_xticks(range(len(signature_feats)))
    ax.set_xticklabels([shorten(f, 26) for f in signature_feats],
                       rotation=70, ha='right', fontsize=8)
    plt.colorbar(im, ax=ax, label='z-score', shrink=0.7)
    ax.set_title('Multi-organelle granularity signature (ER + AGP + MITO, scales 2-4)\n'
                 'HSP90 inhibitors should saturate; STK24 group should not',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'multiorganelle_signature.png', dpi=140, bbox_inches='tight')
    plt.close()
    print(f"  Saved: multiorganelle_signature.png")

# ---------------- Summary text ----------------
print("\nSTEP 7: Write summary")
with open(OUT_DIR / 'summary.txt', 'w', encoding='utf-8') as f:
    f.write("HSP90 vs STK24 — feature-level comparison\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Common features: {len(common_features)}\n")
    f.write(f"Compounds: {len(labels)}\n\n")

    f.write("Compounds in order:\n")
    for lbl, grp, pl in zip(labels, group_labels, plate_labels):
        f.write(f"  {grp:6s}  {lbl:14s}  {pl}\n")
    f.write("\n")

    f.write("PEARSON CORRELATIONS\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'pair':40s}  r\n")
    pair_lines = []
    for i in range(n):
        for j in range(i + 1, n):
            tag = f"{labels[i]:14s} ↔ {labels[j]:14s}"
            r = corr[i, j]
            grp_tag = ''
            if group_labels[i] == group_labels[j]:
                grp_tag = f'  [intra-{group_labels[i]}]'
            else:
                grp_tag = '  [HSP90<->STK24]'
            pair_lines.append((r, f"  {tag}  r = {r:+.3f}{grp_tag}\n"))
    pair_lines.sort(key=lambda x: -x[0])
    for _, line in pair_lines:
        f.write(line)
    f.write("\n")

    f.write(f"  Mean intra-HSP90 r: {np.mean([corr[i, j] for i in hsp_idx for j in hsp_idx if i < j]):.3f}\n")
    f.write(f"  Mean intra-STK24 r: {np.mean([corr[i, j] for i in stk_idx for j in stk_idx if i < j]):.3f}\n")
    f.write(f"  Mean cross HSP90-STK24 r: "
            f"{np.mean([corr[i, j] for i in hsp_idx for j in stk_idx]):.3f}\n")
    f.write("\n")

    f.write("TOP 15 DISCRIMINATING FEATURES (HSP90 mean - STK24 mean)\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'rank':>4}  {'feature':50s}  {'HSP90':>7}  {'STK24':>7}  {'diff':>7}\n")
    for rank, idx in enumerate(top_idx, 1):
        f.write(f"  {rank:2d}  {shorten(common_features[idx], 50):50s}  "
                f"{hsp_mean[idx]:+7.2f}  {stk_mean[idx]:+7.2f}  {diff[idx]:+7.2f}\n")
    f.write("\n")

    f.write("OUTPUTS\n")
    f.write("-" * 70 + "\n")
    for p in sorted(OUT_DIR.glob('*')):
        if p.is_file():
            f.write(f"  {p.name}  ({p.stat().st_size // 1024} KB)\n")

print(f"  Saved: summary.txt")
print(f"\nDONE. All outputs in: {OUT_DIR}")
