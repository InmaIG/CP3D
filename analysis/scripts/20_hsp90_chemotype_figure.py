"""
20_hsp90_chemotype_figure.py
==============================
Publication-ready figures illustrating chemotype-specific recovery of HSP
family inhibitors in 3D HepG2 Cell Painting.

Generates FOUR INDEPENDENT FIGURES (one panel per file) so they can be
combined freely in the final manuscript layout without label overlap:

  panel_A_2d_vs_3d_scatter.{png,pdf,svg}
  panel_B_activity_by_chemotype.{png,pdf,svg}
  panel_C_distance_heatmap.{png,pdf,svg}
  panel_D_2d_vs_3d_ranks_bars.{png,pdf,svg}

Scientific summary
------------------
3 of 4 ansamycins (geldanamycin, retaspimycin, alvespimycin) are robust 3D
hits while the redox-attenuated ansamycin tanespimycin and 6 non-ansamycin
HSP90 inhibitors are not. Pattern consistent with the pipeline capturing
quinone-driven hepatotoxic redox cycling rather than HSP90 inhibition per se.

Inputs
------
results/medina_2d_vs_3d/per_compound.csv
data/processed/consensus_C238[6,7,8].csv
data/annotated/cp3d_library_annotated.csv

Outputs (results/hsp90_chemotype/)
----------------------------------
4 independent panels + hsp90_chemotype_table.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.spatial.distance import pdist, squareform

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 11

BASE = Path(r'C:\Users\Ianezi\Documents\CP3D')
ANALYSIS = BASE / 'analysis'
PER_COMPOUND = ANALYSIS / 'results' / 'medina_2d_vs_3d' / 'per_compound.csv'
PROC = ANALYSIS / 'data' / 'processed'
OUT = ANALYSIS / 'results' / 'hsp90_chemotype'
OUT.mkdir(parents=True, exist_ok=True)

CONSENSUS_FILES = {
    'C2386': PROC / 'consensus_C2386.csv',
    'C2387': PROC / 'consensus_C2387.csv',
    'C2388': PROC / 'consensus_C2388.csv',
}

HSP_COMPOUNDS = {
    'GELDANAMYCIN':  {'chemotype': 'Ansamycin (HSP90)',  'redox': 'high'},
    'ALVESPIMYCIN':  {'chemotype': 'Ansamycin (HSP90)',  'redox': 'high'},
    'RETASPIMYCIN':  {'chemotype': 'Ansamycin (HSP90)',  'redox': 'medium-high'},
    'TANESPIMYCIN':  {'chemotype': 'Ansamycin (HSP90)',  'redox': 'attenuated'},
    'LUMINESPIB':    {'chemotype': 'Resorcinol (HSP90)', 'redox': 'low'},
    'GANETESPIB':    {'chemotype': 'Resorcinol (HSP90)', 'redox': 'low'},
    'VER-49009':     {'chemotype': 'Resorcinol (HSP90)', 'redox': 'low'},
    'BIIB021':       {'chemotype': 'Purine (HSP90)',     'redox': 'low'},
    'SNX-2112':      {'chemotype': 'Benzamide (HSP90)',  'redox': 'low'},
    'VER 155008':    {'chemotype': 'HSP70 inhibitor',    'redox': 'low'},
}

CHEMOTYPE_PALETTE = {
    'Ansamycin (HSP90)':  '#D7263D',
    'Resorcinol (HSP90)': '#1F77B4',
    'Purine (HSP90)':     '#2CA02C',
    'Benzamide (HSP90)':  '#9467BD',
    'HSP70 inhibitor':    '#7F7F7F',
}

def save_in_formats(fig, name):
    """Save figure in PNG, PDF and SVG."""
    for ext in ('png', 'pdf', 'svg'):
        out = OUT / f'{name}.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {name}.{{png,pdf,svg}}")

print(f"\n{'='*70}\n=== HSP chemotype figures (independent panels) ===\n{'='*70}\n")

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print(f"{'-'*70}\n1. Loading data\n{'-'*70}")
pc = pd.read_csv(PER_COMPOUND)
pc['_drug_norm'] = pc['Drug_name'].fillna('').astype(str).str.upper().str.strip()

hsp_rows = []
for compound_name, meta in HSP_COMPOUNDS.items():
    matches = pc[pc['_drug_norm'] == compound_name.upper()]
    if len(matches):
        r = matches.iloc[0]
        hsp_rows.append({
            'Drug_name': compound_name,
            'EOS_id': r['EOS_id'],
            'Chemotype': meta['chemotype'],
            'redox_reactivity': meta['redox'],
            'activity_RMS_3D': r['activity_RMS_3D'],
            'rank_3D_RMS_pct': r['rank_3D_RMS_pct'],
            'activity_RMS_2D': r['activity_RMS_2D'],
            'rank_2D_RMS_pct': r['rank_2D_RMS_pct'],
            'is_hit_3D': bool(r['is_hit_3D']),
        })
    else:
        print(f"  WARNING: {compound_name} not found.")
hsp_df = pd.DataFrame(hsp_rows)
hsp_df = hsp_df.sort_values(['Chemotype', 'rank_3D_RMS_pct'], ascending=[True, False])
hsp_df.to_csv(OUT / 'hsp90_chemotype_table.csv', index=False)
print(f"  HSP inhibitors loaded: {len(hsp_df)}")

# ---------------------------------------------------------------------------
# PANEL A — 2D vs 3D rank scatter (independent figure)
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\nPanel A — 2D vs 3D rank scatter\n{'-'*70}")
fig, ax = plt.subplots(figsize=(10, 8))
for _, r in hsp_df.iterrows():
    color = CHEMOTYPE_PALETTE[r['Chemotype']]
    edgecolor = 'black' if r['is_hit_3D'] else 'white'
    linewidth = 2.0 if r['is_hit_3D'] else 0.8
    marker = '*' if r['is_hit_3D'] else 'o'
    size = 380 if r['is_hit_3D'] else 150
    ax.scatter(r['rank_2D_RMS_pct'], r['rank_3D_RMS_pct'],
                color=color, edgecolor=edgecolor, linewidth=linewidth,
                s=size, marker=marker, zorder=3)

# Annotations with smart offset to avoid overlap
offsets = {
    'GELDANAMYCIN':  (10, 5),
    'ALVESPIMYCIN':  (10, -15),
    'RETASPIMYCIN':  (10, 8),
    'TANESPIMYCIN':  (10, 5),
    'LUMINESPIB':    (10, 5),
    'GANETESPIB':    (-50, -15),
    'VER-49009':     (10, 5),
    'BIIB021':       (10, 5),
    'SNX-2112':      (10, 5),
    'VER 155008':    (10, 5),
}
for _, r in hsp_df.iterrows():
    dx, dy = offsets.get(r['Drug_name'], (10, 5))
    ax.annotate(r['Drug_name'], (r['rank_2D_RMS_pct'], r['rank_3D_RMS_pct']),
                 xytext=(dx, dy), textcoords='offset points', fontsize=11,
                 fontweight='bold' if r['is_hit_3D'] else 'normal',
                 color=CHEMOTYPE_PALETTE[r['Chemotype']])
ax.axhline(95, ls='--', color='#888', lw=0.8, alpha=0.6, label='P95 threshold')
ax.axvline(95, ls='--', color='#888', lw=0.8, alpha=0.6)
ax.set_xlabel('Within-library rank percentile in 2D-MEDINA HepG2', fontsize=13)
ax.set_ylabel('Within-library rank percentile in 3D-MEDINA HepG2', fontsize=13)
ax.set_title('A. Activity rank of HSP family inhibitors in 2D vs 3D MEDINA HepG2 Cell Painting',
              fontsize=13, fontweight='bold', loc='left', pad=15)
ax.set_xlim(-2, 110)
ax.set_ylim(-2, 110)
ax.grid(alpha=0.25)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Chemotype legend
legend_patches = [Patch(facecolor=CHEMOTYPE_PALETTE[c], edgecolor='black', label=c)
                   for c in CHEMOTYPE_PALETTE if c in hsp_df['Chemotype'].unique()]
legend_patches.append(Patch(facecolor='white', edgecolor='black', linewidth=2,
                              label='HIT 3D (star, bold label)'))
ax.legend(handles=legend_patches, loc='lower right', fontsize=10, frameon=True,
           framealpha=0.95)
plt.tight_layout()
save_in_formats(fig, 'panel_A_2d_vs_3d_scatter')
plt.close()

# ---------------------------------------------------------------------------
# PANEL B — Bar chart mean activity per chemotype
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\nPanel B — Activity by chemotype\n{'-'*70}")
fig, ax = plt.subplots(figsize=(10, 7))
chemo_order = ['Ansamycin (HSP90)', 'Resorcinol (HSP90)',
                'HSP70 inhibitor', 'Benzamide (HSP90)', 'Purine (HSP90)']
present_chemo = [c for c in chemo_order if c in hsp_df['Chemotype'].unique()]
group_stats = hsp_df.groupby('Chemotype').agg(
    mean_act=('activity_RMS_3D', 'mean'),
    std_act=('activity_RMS_3D', 'std'),
    n=('Drug_name', 'count')
).reindex(present_chemo)
group_stats['std_act'] = group_stats['std_act'].fillna(0)

x_pos = np.arange(len(group_stats))
bar_colors = [CHEMOTYPE_PALETTE[c] for c in group_stats.index]
ax.bar(x_pos, group_stats['mean_act'], yerr=group_stats['std_act'],
        color=bar_colors, edgecolor='black', linewidth=1.5, capsize=6,
        alpha=0.85, error_kw={'lw': 1.5})

# Overlay individual points with compound names
for i, ct in enumerate(group_stats.index):
    sub = hsp_df[hsp_df['Chemotype'] == ct].sort_values('activity_RMS_3D')
    jitter = (np.arange(len(sub)) - (len(sub)-1)/2) * 0.12
    for j, (_, r) in enumerate(sub.iterrows()):
        marker = '*' if r['is_hit_3D'] else 'o'
        size = 200 if r['is_hit_3D'] else 100
        ax.scatter(i + jitter[j], r['activity_RMS_3D'],
                    color='white', edgecolor='black', s=size, zorder=4,
                    marker=marker, linewidth=1.2)
        ax.annotate(r['Drug_name'],
                     (i + jitter[j], r['activity_RMS_3D']),
                     xytext=(8, 0), textcoords='offset points', fontsize=8.5,
                     fontweight='bold' if r['is_hit_3D'] else 'normal')

ax.set_xticks(x_pos)
ax.set_xticklabels([f'{c}\n(n={int(group_stats.loc[c, "n"])})' for c in group_stats.index],
                     fontsize=11)
ax.set_ylabel('Mean activity_RMS_3D ± SD', fontsize=13)
ax.set_title('B. 3D HepG2 activity by HSP inhibitor chemotype',
              fontsize=13, fontweight='bold', loc='left', pad=15)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(alpha=0.25, axis='y')
ax.set_ylim(0, hsp_df['activity_RMS_3D'].max() * 1.2)
plt.tight_layout()
save_in_formats(fig, 'panel_B_activity_by_chemotype')
plt.close()

# ---------------------------------------------------------------------------
# Compute CellProfiler distance matrix (for Panel C)
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\nComputing CellProfiler distance matrix\n{'-'*70}")
cp_dfs = []
for plate, p in CONSENSUS_FILES.items():
    if p.exists():
        d = pd.read_csv(p)
        d = d[d['Metadata_Well_type'] == 'compound'].copy()
        d = d.rename(columns={'Metadata_Compound': 'EOS_id'})
        cp_dfs.append(d)
cp_all = pd.concat(cp_dfs, ignore_index=True)
feat_cols = [c for c in cp_all.columns if not c.startswith('Metadata_')
              and c != 'EOS_id' and pd.api.types.is_numeric_dtype(cp_all[c])]
cp_unique = cp_all.groupby('EOS_id')[feat_cols].mean()

order_eos = [e for e in hsp_df['EOS_id'] if e in cp_unique.index]
cp_hsp = cp_unique.loc[order_eos]
hsp_ordered = hsp_df.set_index('EOS_id').loc[order_eos].reset_index()

# Standardize using library-wide mean / std
all_X = cp_unique[feat_cols].values.astype(float)
all_X = np.nan_to_num(all_X, nan=0.0)
all_mu, all_sd = all_X.mean(0), all_X.std(0) + 1e-9
X = cp_hsp.values.astype(float)
X = np.nan_to_num(X, nan=0.0)
Xz = (X - all_mu) / all_sd
D = squareform(pdist(Xz, metric='euclidean'))
print(f"  Distance matrix: {D.shape}")

# ---------------------------------------------------------------------------
# PANEL C — Distance heatmap
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\nPanel C — Distance heatmap\n{'-'*70}")
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(D, cmap='RdYlBu_r', aspect='equal')
names = hsp_ordered['Drug_name'].values
chemos = hsp_ordered['Chemotype'].values
hits = hsp_ordered['is_hit_3D'].values
ax.set_xticks(range(len(names)))
ax.set_yticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(names, fontsize=11)
for i, (name, ct, hit) in enumerate(zip(names, chemos, hits)):
    color = CHEMOTYPE_PALETTE[ct]
    ax.get_yticklabels()[i].set_color(color)
    ax.get_xticklabels()[i].set_color(color)
    if hit:
        ax.get_yticklabels()[i].set_fontweight('bold')
        ax.get_xticklabels()[i].set_fontweight('bold')

vmax = D.max()
for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        v = D[i, j]
        text_color = 'white' if v > vmax * 0.65 else 'black'
        ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                 fontsize=9, color=text_color)

# Draw chemotype block boundaries
chemo_groups = []
prev = None
for i, ct in enumerate(chemos):
    if ct != prev:
        chemo_groups.append(i)
        prev = ct
chemo_groups.append(len(chemos))
for boundary in chemo_groups[1:-1]:
    ax.axhline(boundary - 0.5, color='black', lw=1.5)
    ax.axvline(boundary - 0.5, color='black', lw=1.5)

cbar = plt.colorbar(im, ax=ax, shrink=0.7,
                     label='Euclidean distance (CellProfiler 3D features, z-scored)')
cbar.ax.tick_params(labelsize=10)
ax.set_title('C. Pairwise distances between HSP family inhibitors in 3D CellProfiler space\n'
              '(ordered by chemotype; ansamycin hits shown bold)',
              fontsize=13, fontweight='bold', loc='left', pad=15)
plt.tight_layout()
save_in_formats(fig, 'panel_C_distance_heatmap')
plt.close()

# ---------------------------------------------------------------------------
# PANEL D — 2D vs 3D rank bars per compound
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\nPanel D — 2D vs 3D rank bars\n{'-'*70}")
fig, ax = plt.subplots(figsize=(10, 8))
hsp_for_bar = hsp_df.sort_values('rank_3D_RMS_pct', ascending=True).reset_index(drop=True)
y = np.arange(len(hsp_for_bar))
ax.barh(y - 0.20, hsp_for_bar['rank_3D_RMS_pct'], height=0.40,
         color='#D7263D', edgecolor='black', label='3D-MEDINA rank', linewidth=1)
ax.barh(y + 0.20, hsp_for_bar['rank_2D_RMS_pct'], height=0.40,
         color='#1F77B4', edgecolor='black', label='2D-MEDINA rank', linewidth=1)
# Annotate with rank values
for i, (_, r) in enumerate(hsp_for_bar.iterrows()):
    ax.text(r['rank_3D_RMS_pct'] + 1, i - 0.20,
             f"{r['rank_3D_RMS_pct']:.0f}",
             va='center', fontsize=10, color='#D7263D', fontweight='bold')
    ax.text(r['rank_2D_RMS_pct'] + 1, i + 0.20,
             f"{r['rank_2D_RMS_pct']:.0f}",
             va='center', fontsize=10, color='#1F77B4')

ax.set_yticks(y)
ax.set_yticklabels(hsp_for_bar['Drug_name'], fontsize=11)
for i, (_, r) in enumerate(hsp_for_bar.iterrows()):
    color = CHEMOTYPE_PALETTE[r['Chemotype']]
    ax.get_yticklabels()[i].set_color(color)
    if r['is_hit_3D']:
        ax.get_yticklabels()[i].set_fontweight('bold')

ax.set_xlabel('Within-library rank percentile', fontsize=13)
ax.set_title('D. Per-compound activity rank: 2D vs 3D MEDINA HepG2\n'
              '(compound names coloured by chemotype; 3D hits in bold)',
              fontsize=13, fontweight='bold', loc='left', pad=15)
ax.axvline(95, ls='--', color='#888', lw=0.8, alpha=0.6,
            label='P95 threshold')
ax.set_xlim(0, 115)
ax.legend(loc='lower right', fontsize=11)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(alpha=0.25, axis='x')
plt.tight_layout()
save_in_formats(fig, 'panel_D_2d_vs_3d_ranks_bars')
plt.close()

print(f"\n{'='*70}\nDone. 4 independent panels saved in: {OUT}\n{'='*70}")
print("\nFiles produced (PNG + PDF + SVG each):")
for name in ['panel_A_2d_vs_3d_scatter',
             'panel_B_activity_by_chemotype',
             'panel_C_distance_heatmap',
             'panel_D_2d_vs_3d_ranks_bars']:
    print(f"  {name}.{{png,pdf,svg}}")
print(f"\nTable: hsp90_chemotype_table.csv")
