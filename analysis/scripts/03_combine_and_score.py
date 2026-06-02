"""
03_combine_and_score.py
=======================
Combine multiple replicates into a consensus profile per compound.
Compute replicate correlation, activity score and hit ranking.

Inputs:
- data\\processed\\<folder>_normalized_dmso_selected.csv  (one per replicate)

Outputs:
- data\\processed\\combined_<plate>.csv         (raw combined, one row per spheroid)
- data\\processed\\consensus_<plate>.csv        (aggregated, one row per compound)
- data\\processed\\activity_<plate>.csv         (compound + activity score, ranked)
- results\\multi_rep\\<plate>_replicate_corr.png
- results\\multi_rep\\<plate>_activity_distribution.png
- results\\multi_rep\\<plate>_top_hits_heatmap.png
- results\\multi_rep\\<plate>_summary.txt

Uso:
    python 03_combine_and_score.py --plate C2386 --replicates C2386R1 C2386R2 C2386R3 C2386R4
    python 03_combine_and_score.py --plate C2386 --replicates C2386R1 C2386R2 C2386R3 C2386R4 --hit_threshold 95
"""

import argparse
import sys
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*tick_labels.*')

# ---------------- Paths ----------------
BASE = Path(r'C:\Users\Ianezi\Documents\CP3D\analysis')
PROC_DIR = BASE / 'data' / 'processed'
OUT_DIR = BASE / 'results' / 'multi_rep'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Args ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--plate', required=True,
                    help='Plate base name (e.g. C2386).')
parser.add_argument('--replicates', nargs='+', required=True,
                    help='List of replicate folder names (e.g. C2386R1 C2386R2 C2386R3 C2386R4).')
parser.add_argument('--aggregate_method', default='median',
                    choices=['median', 'mean'],
                    help='How to aggregate replicates per compound (default: median).')
parser.add_argument('--hit_threshold', type=float, default=95,
                    help='Activity percentile to call hits (default: 95 = top 5 percent).')
parser.add_argument('--top_n_heatmap', type=int, default=30,
                    help='Number of top hits to show in heatmap (default: 30).')
args = parser.parse_args()

plate = args.plate
replicates = args.replicates

print(f"\n{'='*70}\n=== COMBINE + SCORE: {plate} ===\n{'='*70}")
print(f"Replicates: {replicates}")
print(f"Aggregate: {args.aggregate_method}")
print(f"Hit threshold: top {100-args.hit_threshold:.0f}% (percentile {args.hit_threshold})")

# ---------------- Load all replicates ----------------
print(f"\n{'-'*70}\nStep 1: Load all replicate CSVs\n{'-'*70}")

dfs = []
features_per_rep = {}
for rep in replicates:
    fp = PROC_DIR / f'{rep}_normalized_dmso_selected.csv'
    if not fp.exists():
        print(f"\nERROR: No se encuentra {fp}")
        print(f"       Ejecuta 02_normalize.py para {rep} primero.")
        sys.exit(1)
    df = pd.read_csv(fp)
    df['Metadata_Replicate'] = rep
    dfs.append(df)
    
    # Identify features in this replicate
    meta = [c for c in df.columns if c.startswith('Metadata_') 
            or c.startswith('FileName_') or c.startswith('PathName_')
            or c in ['ImageNumber', 'ObjectNumber']]
    feats = [c for c in df.columns if c not in meta]
    features_per_rep[rep] = set(feats)
    print(f"  {rep}: {len(df)} rows x {len(feats)} features")

# ---------------- Find common features ----------------
print(f"\n{'-'*70}\nStep 2: Identify common features across replicates\n{'-'*70}")

common_features = set.intersection(*features_per_rep.values())
print(f"  Total unique features across replicates: "
      f"{len(set.union(*features_per_rep.values()))}")
print(f"  Features common to ALL replicates: {len(common_features)}")
for rep, feats in features_per_rep.items():
    only_here = feats - common_features
    print(f"    {rep}: {len(feats)} total, {len(feats)-len(common_features)} replicate-specific")

if len(common_features) < 50:
    print(f"\nWARNING: Only {len(common_features)} common features. "
          f"Replicates may differ too much in feature selection.")

# ---------------- Combine ----------------
print(f"\n{'-'*70}\nStep 3: Combine replicates\n{'-'*70}")

# Keep only common features + metadata
common_features = sorted(common_features)
all_meta_cols = set()
for df in dfs:
    all_meta_cols.update(c for c in df.columns 
                         if c.startswith('Metadata_') 
                         or c in ['ImageNumber', 'ObjectNumber'])
all_meta_cols = sorted(all_meta_cols)

aligned_dfs = []
for df in dfs:
    keep = [c for c in all_meta_cols if c in df.columns] + common_features
    aligned_dfs.append(df[keep].copy())

combined = pd.concat(aligned_dfs, ignore_index=True)
print(f"  Combined shape: {combined.shape}")
print(f"  Total spheroids across replicates: {len(combined)}")
print(f"  By replicate:")
for rep, n in combined['Metadata_Replicate'].value_counts().items():
    print(f"    {rep}: {n}")

# Compounds per replicate
print(f"\n  Compounds detected:")
for rep in replicates:
    sub = combined[combined['Metadata_Replicate']==rep]
    n_compounds = sub[sub['Metadata_Well_type']=='compound']['Metadata_Compound'].nunique()
    print(f"    {rep}: {n_compounds} unique compounds")

out_combined = PROC_DIR / f'combined_{plate}.csv'
combined.to_csv(out_combined, index=False)
print(f"\n  Saved combined: {out_combined}")

# ---------------- Replicate correlation ----------------
print(f"\n{'-'*70}\nStep 4: Replicate correlation\n{'-'*70}")

# For each compound, compute correlation between its replicate profiles
compound_corrs = []  # List of dicts: {compound, n_replicates, mean_corr, profiles...}

for compound, sub in combined[combined['Metadata_Well_type']=='compound'].groupby('Metadata_Compound'):
    if len(sub) < 2:
        continue
    profiles = sub[common_features].values  # shape (n_reps, n_features)
    # Pairwise pearson correlations
    corrs = []
    for i in range(len(profiles)):
        for j in range(i+1, len(profiles)):
            v1, v2 = profiles[i], profiles[j]
            mask = ~(np.isnan(v1) | np.isnan(v2))
            if mask.sum() > 10:
                # pearson
                v1m, v2m = v1[mask], v2[mask]
                if v1m.std() > 0 and v2m.std() > 0:
                    c = np.corrcoef(v1m, v2m)[0,1]
                    corrs.append(c)
    if corrs:
        compound_corrs.append({
            'compound': compound,
            'n_replicates': len(sub),
            'mean_corr': np.mean(corrs),
            'min_corr': np.min(corrs),
            'max_corr': np.max(corrs),
        })

corr_df = pd.DataFrame(compound_corrs)
print(f"  Compounds with >=2 replicates: {len(corr_df)}")
if len(corr_df) > 0:
    print(f"\n  Replicate correlation distribution (Pearson):")
    print(f"    Median: {corr_df['mean_corr'].median():.3f}")
    print(f"    P25-P75: {corr_df['mean_corr'].quantile(0.25):.3f} - {corr_df['mean_corr'].quantile(0.75):.3f}")
    print(f"    P5-P95:  {corr_df['mean_corr'].quantile(0.05):.3f} - {corr_df['mean_corr'].quantile(0.95):.3f}")

# Same for DMSO (sanity: DMSO replicates should correlate moderately, not perfectly)
dmso_corrs = []
dmso_sub = combined[combined['Metadata_Compound']=='DMSO']
if len(dmso_sub) > 1:
    profiles = dmso_sub[common_features].values
    for i in range(len(profiles)):
        for j in range(i+1, len(profiles)):
            v1, v2 = profiles[i], profiles[j]
            mask = ~(np.isnan(v1) | np.isnan(v2))
            if mask.sum() > 10 and v1[mask].std() > 0 and v2[mask].std() > 0:
                c = np.corrcoef(v1[mask], v2[mask])[0,1]
                dmso_corrs.append(c)
print(f"\n  DMSO inter-well correlation: median={np.median(dmso_corrs):.3f} "
      f"(n={len(dmso_corrs)} pairs) - should be near 0 since centered/scaled")

# Plot replicate correlation distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
if len(corr_df) > 0:
    ax.hist(corr_df['mean_corr'], bins=50, color='#3498DB', edgecolor='black', alpha=0.7)
    ax.axvline(corr_df['mean_corr'].median(), color='red', linestyle='--', 
               label=f"median={corr_df['mean_corr'].median():.3f}")
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Mean replicate Pearson correlation')
    ax.set_ylabel('Number of compounds')
    ax.set_title(f'Replicate correlation per compound\n(n={len(corr_df)} compounds, '
                 f'{len(replicates)} replicates each)')
    ax.legend()
    ax.grid(alpha=0.3)

ax = axes[1]
if len(corr_df) > 0:
    # Strong reproducibility: corr > 0.5; low: corr < 0.2
    strong = (corr_df['mean_corr'] > 0.5).sum()
    moderate = ((corr_df['mean_corr'] > 0.2) & (corr_df['mean_corr'] <= 0.5)).sum()
    weak = (corr_df['mean_corr'] <= 0.2).sum()
    cats = ['Weak\n(< 0.2)', 'Moderate\n(0.2-0.5)', 'Strong\n(> 0.5)']
    counts = [weak, moderate, strong]
    colors = ['#E74C3C', '#F39C12', '#27AE60']
    bars = ax.bar(cats, counts, color=colors, alpha=0.8, edgecolor='black')
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{n}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of compounds')
    ax.set_title('Reproducibility categories')
    ax.grid(alpha=0.3, axis='y')

plt.suptitle(f'Replicate reproducibility: {plate}', fontsize=12, fontweight='bold')
plt.tight_layout()
corr_plot = OUT_DIR / f'{plate}_replicate_corr.png'
plt.savefig(corr_plot, dpi=120, bbox_inches='tight')
print(f"\n  Plot saved: {corr_plot}")
plt.close()

# ---------------- Aggregate to consensus profiles ----------------
print(f"\n{'-'*70}\nStep 5: Aggregate to consensus per compound\n{'-'*70}")

# For each compound, aggregate features across replicates
agg_func = np.median if args.aggregate_method == 'median' else np.mean

# Group: by Compound (aggregating across replicates)
# For each compound, also store Well_type and tracking info
metadata_for_agg = combined.groupby('Metadata_Compound').agg({
    'Metadata_Well_type': 'first',
    'Metadata_Plate': 'first',
    'Metadata_Replicate': lambda x: ','.join(sorted(x.unique())),
}).reset_index()
metadata_for_agg.columns = ['Metadata_Compound', 'Metadata_Well_type',
                             'Metadata_Plate', 'Metadata_Replicates_used']

# n_replicates per compound
n_reps = combined.groupby('Metadata_Compound').size().reset_index(name='Metadata_n_replicates')
metadata_for_agg = metadata_for_agg.merge(n_reps, on='Metadata_Compound')

# Aggregate features
features_agg = combined.groupby('Metadata_Compound')[common_features].apply(
    lambda g: g.apply(agg_func, axis=0)
).reset_index()

consensus = metadata_for_agg.merge(features_agg, on='Metadata_Compound')

print(f"  Consensus profiles: {len(consensus)} compounds")
print(f"  By type:")
for t, n in consensus['Metadata_Well_type'].value_counts().items():
    print(f"    {t}: {n}")
print(f"  Replicate coverage:")
for n, count in consensus['Metadata_n_replicates'].value_counts().sort_index().items():
    print(f"    {n} replicate(s): {count} compounds")

out_consensus = PROC_DIR / f'consensus_{plate}.csv'
consensus.to_csv(out_consensus, index=False)
print(f"\n  Saved consensus: {out_consensus}")

# ---------------- Activity score ----------------
print(f"\n{'-'*70}\nStep 6: Activity score (distance from DMSO)\n{'-'*70}")

# Activity = RMS of consensus profile (root mean square of features)
# A consensus profile of pure noise has RMS ~ 1 (because each feature is ~ N(0, 1))
# A truly active compound should have RMS > 1

profile_matrix = consensus[common_features].fillna(0).values
n_features = profile_matrix.shape[1]
activity = np.sqrt((profile_matrix ** 2).sum(axis=1) / n_features)
consensus['Metadata_Activity'] = activity

# Compare DMSO activity to compound activity
dmso_activity = consensus[consensus['Metadata_Compound']=='DMSO']['Metadata_Activity']
cpd_activity = consensus[consensus['Metadata_Well_type']=='compound']['Metadata_Activity']

print(f"  DMSO consensus activity:     {dmso_activity.values[0]:.3f}")
print(f"    (should be near 0 since DMSO is the reference)")
print(f"  Compound activity stats:")
print(f"    Median:      {cpd_activity.median():.3f}")
print(f"    P25 - P75:   {cpd_activity.quantile(0.25):.3f} - {cpd_activity.quantile(0.75):.3f}")
print(f"    P5  - P95:   {cpd_activity.quantile(0.05):.3f} - {cpd_activity.quantile(0.95):.3f}")
print(f"    Max:         {cpd_activity.max():.3f}")

# ---------------- Hit calling ----------------
print(f"\n{'-'*70}\nStep 7: Hit calling\n{'-'*70}")

threshold_value = cpd_activity.quantile(args.hit_threshold / 100)
print(f"  Threshold (P{args.hit_threshold:.0f} of compounds): activity = {threshold_value:.3f}")

# Filter compounds and rank
compounds_only = consensus[consensus['Metadata_Well_type']=='compound'].copy()
compounds_only = compounds_only.sort_values('Metadata_Activity', ascending=False)
compounds_only['Metadata_Hit'] = compounds_only['Metadata_Activity'] >= threshold_value
compounds_only['Metadata_Rank'] = range(1, len(compounds_only)+1)

n_hits = compounds_only['Metadata_Hit'].sum()
print(f"  Hits identified: {n_hits} compounds ({100*n_hits/len(compounds_only):.1f}%)")

# AreaShape directionality
if 'AreaShape_Area' in compounds_only.columns:
    hits = compounds_only[compounds_only['Metadata_Hit']]
    pos_area = (hits['AreaShape_Area'] > 1).sum()  # expanded
    neg_area = (hits['AreaShape_Area'] < -1).sum()  # shrunken
    print(f"\n  Among hits:")
    print(f"    With AreaShape_Area > +1 (expanded-like): {pos_area}")
    print(f"    With AreaShape_Area < -1 (shrunken): {neg_area}")
    print(f"    Other: {n_hits - pos_area - neg_area}")

# Save activity table (compound, replicates_used, n_replicates, activity, hit, rank, area)
activity_cols = ['Metadata_Rank', 'Metadata_Compound', 'Metadata_Activity',
                 'Metadata_Hit', 'Metadata_n_replicates', 'Metadata_Replicates_used']
if 'AreaShape_Area' in compounds_only.columns:
    activity_cols.append('AreaShape_Area')

activity_table = compounds_only[activity_cols].copy()
out_activity = PROC_DIR / f'activity_{plate}.csv'
activity_table.to_csv(out_activity, index=False)
print(f"\n  Saved activity table: {out_activity}")

# Print top 20
print(f"\n  Top 20 most active compounds:")
print(activity_table.head(20).to_string(index=False))

# ---------------- Plot: activity distribution ----------------
print(f"\n{'-'*70}\nStep 8: Visualizations\n{'-'*70}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(cpd_activity, bins=50, color='#3498DB', edgecolor='black', alpha=0.7,
        label=f'Compounds (n={len(cpd_activity)})')
ax.axvline(threshold_value, color='red', linestyle='--', 
           label=f'Hit threshold (P{args.hit_threshold:.0f}={threshold_value:.2f})')
ax.axvline(dmso_activity.values[0], color='green', linestyle='-',
           label=f'DMSO (={dmso_activity.values[0]:.3f})')
ax.set_xlabel('Activity score (RMS of consensus profile)')
ax.set_ylabel('Number of compounds')
ax.set_title('Activity score distribution')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
# Volcano-like: activity vs AreaShape_Area
if 'AreaShape_Area' in compounds_only.columns:
    cpd = compounds_only.copy()
    colors = ['#95A5A6'] * len(cpd)
    for i, (idx, row) in enumerate(cpd.iterrows()):
        if row['Metadata_Hit']:
            if row['AreaShape_Area'] > 1:
                colors[i] = '#E67E22'  # expanded
            elif row['AreaShape_Area'] < -1:
                colors[i] = '#9B59B6'  # shrunken
            else:
                colors[i] = '#E74C3C'  # other hit
    ax.scatter(cpd['AreaShape_Area'], cpd['Metadata_Activity'],
               c=colors, alpha=0.6, s=20, edgecolors='black', linewidths=0.3)
    ax.axhline(threshold_value, color='red', linestyle='--', alpha=0.5,
               label=f'Hit threshold')
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('AreaShape_Area (normalized)')
    ax.set_ylabel('Activity score')
    ax.set_title('Activity vs Area: phenotype landscape')
    
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#95A5A6', markersize=8, label='Inactive'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#E67E22', markersize=8, label='Hit + expanded (Area >> 0)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#9B59B6', markersize=8, label='Hit + compact toxic (Area << 0)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#E74C3C', markersize=8, label='Hit (other)'),
    ]
    ax.legend(handles=legend, loc='best', fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle(f'Activity analysis: {plate}', fontsize=12, fontweight='bold')
plt.tight_layout()
act_plot = OUT_DIR / f'{plate}_activity_distribution.png'
plt.savefig(act_plot, dpi=120, bbox_inches='tight')
print(f"  Activity plot saved: {act_plot}")
plt.close()

# ---------------- Heatmap of top hits ----------------
top_n = min(args.top_n_heatmap, n_hits) if n_hits > 0 else min(args.top_n_heatmap, len(compounds_only))
top_hits = compounds_only.head(top_n)

# Pick representative features (most variable across hits)
feature_var = top_hits[common_features].var(axis=0)
top_features = feature_var.nlargest(40).index.tolist()

heat_data = top_hits[top_features].values
heat_data = np.clip(heat_data, -5, 5)

fig, ax = plt.subplots(figsize=(14, max(6, top_n*0.3)))
cmap = LinearSegmentedColormap.from_list('rwb', ['#2E75B6', 'white', '#C00000'])
im = ax.imshow(heat_data, cmap=cmap, vmin=-5, vmax=5, aspect='auto')
ax.set_yticks(range(len(top_hits)))
ax.set_yticklabels(top_hits['Metadata_Compound'].values, fontsize=8)
ax.set_xticks(range(len(top_features)))
ax.set_xticklabels([f[:30] for f in top_features], rotation=90, fontsize=7)
ax.set_title(f'Top {top_n} most active compounds — top 40 most variable features\n'
             f'Blue = below DMSO, Red = above DMSO (clipped at +/-5 MADs)',
             fontsize=11)
plt.colorbar(im, ax=ax, label='Normalized value (MAD units)')
plt.tight_layout()
heat_plot = OUT_DIR / f'{plate}_top_hits_heatmap.png'
plt.savefig(heat_plot, dpi=120, bbox_inches='tight')
print(f"  Heatmap saved: {heat_plot}")
plt.close()

# ---------------- Summary ----------------
summary_path = OUT_DIR / f'{plate}_summary.txt'
with open(summary_path, 'w') as f:
    f.write(f"Multi-replicate analysis: {plate}\n")
    f.write(f"="*70 + "\n\n")
    f.write(f"Replicates: {', '.join(replicates)}\n")
    f.write(f"Aggregate method: {args.aggregate_method}\n\n")
    f.write(f"Spheroids by replicate:\n")
    for rep, n in combined['Metadata_Replicate'].value_counts().items():
        f.write(f"  {rep}: {n}\n")
    f.write(f"\nFeatures common to all replicates: {len(common_features)}\n")
    f.write(f"\nConsensus profiles: {len(consensus)} compounds\n")
    f.write(f"  By type:\n")
    for t, n in consensus['Metadata_Well_type'].value_counts().items():
        f.write(f"    {t}: {n}\n")
    f.write(f"\nReplicate correlation:\n")
    if len(corr_df) > 0:
        f.write(f"  Median: {corr_df['mean_corr'].median():.3f}\n")
        f.write(f"  P25-P75: {corr_df['mean_corr'].quantile(0.25):.3f} - {corr_df['mean_corr'].quantile(0.75):.3f}\n")
        strong = (corr_df['mean_corr'] > 0.5).sum()
        moderate = ((corr_df['mean_corr'] > 0.2) & (corr_df['mean_corr'] <= 0.5)).sum()
        weak = (corr_df['mean_corr'] <= 0.2).sum()
        f.write(f"  Strong (>0.5): {strong}, Moderate (0.2-0.5): {moderate}, Weak (<0.2): {weak}\n")
    f.write(f"\nActivity score:\n")
    f.write(f"  DMSO consensus: {dmso_activity.values[0]:.3f}\n")
    f.write(f"  Compounds median: {cpd_activity.median():.3f}\n")
    f.write(f"  Compounds max: {cpd_activity.max():.3f}\n")
    f.write(f"\nHit threshold (P{args.hit_threshold:.0f}): {threshold_value:.3f}\n")
    f.write(f"Hits: {n_hits} compounds\n")
    if 'AreaShape_Area' in compounds_only.columns:
        hits = compounds_only[compounds_only['Metadata_Hit']]
        pos_area = (hits['AreaShape_Area'] > 1).sum()
        neg_area = (hits['AreaShape_Area'] < -1).sum()
        f.write(f"  Expanded (Area > +1): {pos_area}\n")
        f.write(f"  Cytotoxic compact (Area < -1): {neg_area}\n")
        f.write(f"  Other: {n_hits - pos_area - neg_area}\n")
    f.write(f"\nOutputs:\n")
    f.write(f"  Combined: combined_{plate}.csv\n")
    f.write(f"  Consensus: consensus_{plate}.csv\n")
    f.write(f"  Activity ranking: activity_{plate}.csv\n")

print(f"  Summary saved: {summary_path}")

print(f"\n{'='*70}\n=== DONE ===\n{'='*70}")
print(f"\nKey outputs:")
print(f"  - Activity ranking: {out_activity}")
print(f"  - Top hits heatmap: {heat_plot}")
print(f"  - Activity plot: {act_plot}")
print(f"  - Summary: {summary_path}")