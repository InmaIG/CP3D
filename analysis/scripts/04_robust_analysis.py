"""
04_robust_analysis.py (v2 — preserves key features through feature selection)
=============================================================================
CHANGES v2:
- ALWAYS_KEEP_FEATURES: features that survive feature_select unconditionally,
  so phenotype classification (expanded vs shrunken) always works.

Uso:
    python 04_robust_analysis.py --plate C2386 --replicates C2386R1 C2386R2 C2386R3 C2386R4
"""

import argparse
import sys
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from pycytominer import feature_select

BASE = Path(r'C:\Users\Ianezi\Documents\CP3D\analysis')
PROC_DIR = BASE / 'data' / 'processed'
OUT_DIR = BASE / 'results' / 'multi_rep'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Critical features that must always survive feature_select
ALWAYS_KEEP_FEATURES = ['AreaShape_Area']

parser = argparse.ArgumentParser()
parser.add_argument('--plate', required=True)
parser.add_argument('--replicates', nargs='+', required=True)
parser.add_argument('--corr_threshold', type=float, default=0.9)
parser.add_argument('--activity_threshold_pct', type=float, default=90)
parser.add_argument('--rep_corr_threshold', type=float, default=0.3)
parser.add_argument('--final_hit_pct', type=float, default=95)
args = parser.parse_args()

plate = args.plate
replicates = args.replicates

print(f"\n{'='*70}\n=== ROBUST ANALYSIS: {plate} ===\n{'='*70}")
print(f"Replicates: {replicates}")
print(f"Activity candidates threshold: top {100-args.activity_threshold_pct:.0f}%")
print(f"Replicate correlation threshold: {args.rep_corr_threshold}")
print(f"Final hit threshold: top {100-args.final_hit_pct:.0f}%")
print(f"ALWAYS_KEEP_FEATURES: {ALWAYS_KEEP_FEATURES}")

# Step 1: Load
print(f"\n{'-'*70}\nStep 1: Load normalized data (pre-feature-selection)\n{'-'*70}")
dfs = []
for rep in replicates:
    fp = PROC_DIR / f'{rep}_normalized_dmso.csv'
    if not fp.exists():
        print(f"\nERROR: No se encuentra {fp}")
        sys.exit(1)
    df = pd.read_csv(fp)
    df['Metadata_Replicate'] = rep
    dfs.append(df)
    meta = [c for c in df.columns if c.startswith('Metadata_') 
            or c.startswith('FileName_') or c.startswith('PathName_')
            or c in ['ImageNumber', 'ObjectNumber']]
    feats = [c for c in df.columns if c not in meta]
    print(f"  {rep}: {len(df)} rows x {len(feats)} features (pre-selection)")

# Step 2: Common features
print(f"\n{'-'*70}\nStep 2: Identify common features across replicates\n{'-'*70}")
features_per_rep = {}
for df in dfs:
    rep = df['Metadata_Replicate'].iloc[0]
    meta = [c for c in df.columns if c.startswith('Metadata_') 
            or c.startswith('FileName_') or c.startswith('PathName_')
            or c in ['ImageNumber', 'ObjectNumber']]
    feats = [c for c in df.columns if c not in meta]
    features_per_rep[rep] = set(feats)

common_features = sorted(set.intersection(*features_per_rep.values()))
print(f"  Common features (passed MAD filter in ALL replicates): {len(common_features)}")
for rep, feats in features_per_rep.items():
    print(f"    {rep}: {len(feats)} features ({len(feats - set(common_features))} not in common set)")

# Check always-keep features status
common_set = set(common_features)
missing_always_keep = [f for f in ALWAYS_KEEP_FEATURES if f not in common_set]
if missing_always_keep:
    print(f"\n  WARNING: Always-keep features NOT in common features (failed MAD in some rep): {missing_always_keep}")
    # Try to add them by checking each replicate individually
    for f in missing_always_keep:
        in_reps = [rep for rep, feats in features_per_rep.items() if f in feats]
        print(f"    {f} present in: {in_reps}")
        if in_reps:
            print(f"    Will force-add (filling missing reps with NaN)")
            common_features.append(f)
            common_set.add(f)

# Step 3: Combine + feature_select
print(f"\n{'-'*70}\nStep 3: Combine and apply GLOBAL feature selection\n{'-'*70}")
all_meta_cols = set()
for df in dfs:
    all_meta_cols.update(c for c in df.columns 
                         if c.startswith('Metadata_') or c in ['ImageNumber', 'ObjectNumber'])
all_meta_cols = sorted(all_meta_cols)

aligned_dfs = []
for df in dfs:
    keep = [c for c in all_meta_cols if c in df.columns] + [f for f in common_features if f in df.columns]
    sub = df[keep].copy()
    # Add missing common features with NaN if necessary
    for f in common_features:
        if f not in sub.columns:
            sub[f] = np.nan
    aligned_dfs.append(sub)
combined_pre = pd.concat(aligned_dfs, ignore_index=True)
print(f"  Combined (pre-selection): {combined_pre.shape}")

print(f"\n  Applying global feature_select...")
selected_combined = feature_select(
    profiles=combined_pre,
    features=common_features,
    operation=['variance_threshold', 'correlation_threshold',
               'drop_na_columns', 'blocklist'],
    na_cutoff=0.05,
    corr_threshold=args.corr_threshold,
    freq_cut=0.01,
    unique_cut=0.1,
)

global_features = [c for c in selected_combined.columns if c in common_features]
print(f"  Features before global selection: {len(common_features)}")
print(f"  Features after global selection:  {len(global_features)}")

# CRITICAL: Re-add always-keep features if removed
removed_always_keep = [f for f in ALWAYS_KEEP_FEATURES 
                      if f in common_set and f not in global_features]
if removed_always_keep:
    print(f"\n  *** RESTORING always-keep features removed by selection: {removed_always_keep} ***")
    for f in removed_always_keep:
        selected_combined[f] = combined_pre[f].values
        global_features.append(f)
    print(f"  Final feature count: {len(global_features)}")

print(f"  Removed: {len(common_features) - len(global_features)} "
      f"({100*(len(common_features)-len(global_features))/len(common_features):.1f}%)")

combined = selected_combined.copy()
out_combined = PROC_DIR / f'global_combined_{plate}.csv'
combined.to_csv(out_combined, index=False)
print(f"\n  Saved: {out_combined}")

# Step 4: Aggregate
print(f"\n{'-'*70}\nStep 4: Aggregate to consensus per compound\n{'-'*70}")
metadata_for_agg = combined.groupby('Metadata_Compound').agg({
    'Metadata_Well_type': 'first',
    'Metadata_Plate': 'first',
    'Metadata_Replicate': lambda x: ','.join(sorted(x.unique())),
}).reset_index()
metadata_for_agg.columns = ['Metadata_Compound', 'Metadata_Well_type',
                             'Metadata_Plate', 'Metadata_Replicates_used']
n_reps = combined.groupby('Metadata_Compound').size().reset_index(name='Metadata_n_replicates')
metadata_for_agg = metadata_for_agg.merge(n_reps, on='Metadata_Compound')

features_agg = combined.groupby('Metadata_Compound')[global_features].apply(
    lambda g: g.apply(np.median, axis=0)
).reset_index()
consensus = metadata_for_agg.merge(features_agg, on='Metadata_Compound')

# Activity score
profile_matrix = consensus[global_features].fillna(0).values
n_features = profile_matrix.shape[1]
activity = np.sqrt((profile_matrix ** 2).sum(axis=1) / n_features)
consensus['Metadata_Activity'] = activity

print(f"  Consensus profiles: {len(consensus)} compounds")
cpd = consensus[consensus['Metadata_Well_type']=='compound']
print(f"  Activity stats:")
print(f"    DMSO: {consensus[consensus['Metadata_Compound']=='DMSO']['Metadata_Activity'].values[0]:.3f}")
print(f"    Compounds median: {cpd['Metadata_Activity'].median():.3f}")
print(f"    Compounds max:    {cpd['Metadata_Activity'].max():.3f}")

# Step 5: Active candidates
print(f"\n{'-'*70}\nStep 5: Identify active candidates (top {100-args.activity_threshold_pct:.0f}%)\n{'-'*70}")
candidate_threshold = cpd['Metadata_Activity'].quantile(args.activity_threshold_pct / 100)
print(f"  Activity threshold: {candidate_threshold:.3f}")
active_candidates = consensus[
    (consensus['Metadata_Well_type']=='compound') &
    (consensus['Metadata_Activity'] >= candidate_threshold)
]['Metadata_Compound'].tolist()
print(f"  Active candidates: {len(active_candidates)} compounds")

# Step 6: Replicate correlation
print(f"\n{'-'*70}\nStep 6: Replicate correlation (global vs active-only)\n{'-'*70}")

def compute_rep_corrs(combined_df, compound_list, features):
    results = []
    sub_all = combined_df[combined_df['Metadata_Compound'].isin(compound_list)]
    for compound, sub in sub_all.groupby('Metadata_Compound'):
        if len(sub) < 2:
            continue
        profiles = sub[features].values
        corrs = []
        for i in range(len(profiles)):
            for j in range(i+1, len(profiles)):
                v1, v2 = profiles[i], profiles[j]
                mask = ~(np.isnan(v1) | np.isnan(v2))
                if mask.sum() > 10:
                    v1m, v2m = v1[mask], v2[mask]
                    if v1m.std() > 0 and v2m.std() > 0:
                        c = np.corrcoef(v1m, v2m)[0,1]
                        corrs.append(c)
        if corrs:
            results.append({
                'compound': compound,
                'n_replicates': len(sub),
                'mean_corr': np.mean(corrs),
            })
    return pd.DataFrame(results)

all_compounds = combined[combined['Metadata_Well_type']=='compound']['Metadata_Compound'].unique().tolist()
all_corr_df = compute_rep_corrs(combined, all_compounds, global_features)
active_corr_df = compute_rep_corrs(combined, active_candidates, global_features)

print(f"\n  ALL compounds (n={len(all_corr_df)}):")
print(f"    Median correlation: {all_corr_df['mean_corr'].median():.3f}")
print(f"    P25-P75: {all_corr_df['mean_corr'].quantile(0.25):.3f} - {all_corr_df['mean_corr'].quantile(0.75):.3f}")

if len(active_corr_df) > 0:
    print(f"\n  ACTIVE candidates only (n={len(active_corr_df)}):")
    print(f"    Median correlation: {active_corr_df['mean_corr'].median():.3f}")
    print(f"    P25-P75: {active_corr_df['mean_corr'].quantile(0.25):.3f} - {active_corr_df['mean_corr'].quantile(0.75):.3f}")
    print(f"    P5-P95:  {active_corr_df['mean_corr'].quantile(0.05):.3f} - {active_corr_df['mean_corr'].quantile(0.95):.3f}")

# Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.hist(all_corr_df['mean_corr'], bins=50, color='#95A5A6', alpha=0.7,
        edgecolor='black', label=f'All (n={len(all_corr_df)})')
if len(active_corr_df) > 0:
    ax.hist(active_corr_df['mean_corr'], bins=30, color='#E74C3C', alpha=0.7,
            edgecolor='black', label=f'Active (n={len(active_corr_df)})')
ax.axvline(args.rep_corr_threshold, color='green', linestyle='-',
           label=f'Hit threshold ({args.rep_corr_threshold})')
ax.set_xlabel('Mean replicate Pearson correlation')
ax.set_ylabel('Number of compounds')
ax.set_title('Replicate reproducibility')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
if len(active_corr_df) > 0:
    corr_with_activity = active_corr_df.merge(
        cpd[['Metadata_Compound', 'Metadata_Activity']],
        left_on='compound', right_on='Metadata_Compound')
    ax.scatter(corr_with_activity['Metadata_Activity'],
               corr_with_activity['mean_corr'],
               c='#E74C3C', alpha=0.6, s=30, edgecolors='black', linewidths=0.3)
    ax.axhline(args.rep_corr_threshold, color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Activity score')
    ax.set_ylabel('Replicate correlation')
    ax.set_title('Activity vs Reproducibility')
    ax.grid(alpha=0.3)

plt.suptitle(f'Replicate reproducibility: {plate}', fontsize=12, fontweight='bold')
plt.tight_layout()
plot1 = OUT_DIR / f'{plate}_global_repcorr.png'
plt.savefig(plot1, dpi=120, bbox_inches='tight')
print(f"\n  Plot saved: {plot1}")
plt.close()

# Step 7: Confirmed hits
print(f"\n{'-'*70}\nStep 7: Confirmed hits (active AND reproducible)\n{'-'*70}")
final_threshold = cpd['Metadata_Activity'].quantile(args.final_hit_pct / 100)
print(f"  Final activity threshold (P{args.final_hit_pct:.0f}): {final_threshold:.3f}")

all_corr_dict = dict(zip(active_corr_df['compound'], active_corr_df['mean_corr'])) if len(active_corr_df) > 0 else {}
hits_data = []
for _, row in cpd.iterrows():
    c = row['Metadata_Compound']
    activity_val = row['Metadata_Activity']
    rep_corr = all_corr_dict.get(c, np.nan)
    is_active = activity_val >= final_threshold
    is_reproducible = (not np.isnan(rep_corr)) and rep_corr >= args.rep_corr_threshold
    is_confirmed = is_active and is_reproducible
    area = row.get('AreaShape_Area', np.nan)
    hits_data.append({
        'Metadata_Compound': c,
        'Metadata_Activity': activity_val,
        'Metadata_Replicate_corr': rep_corr,
        'Metadata_n_replicates': row['Metadata_n_replicates'],
        'Metadata_Replicates_used': row['Metadata_Replicates_used'],
        'AreaShape_Area_consensus': area,
        'is_active_top5pct': is_active,
        'is_reproducible': is_reproducible,
        'is_confirmed_hit': is_confirmed,
    })

hits_df = pd.DataFrame(hits_data).sort_values('Metadata_Activity', ascending=False).reset_index(drop=True)
hits_df['Metadata_Rank'] = range(1, len(hits_df)+1)

n_active = hits_df['is_active_top5pct'].sum()
n_reproducible = hits_df['is_reproducible'].sum()
n_confirmed = hits_df['is_confirmed_hit'].sum()
print(f"\n  Compounds in top {100-args.final_hit_pct:.0f}% activity:    {n_active}")
print(f"  Compounds with corr > {args.rep_corr_threshold}:    {n_reproducible}")
print(f"  CONFIRMED HITS (both):              {n_confirmed}")

if 'AreaShape_Area_consensus' in hits_df.columns:
    confirmed = hits_df[hits_df['is_confirmed_hit']]
    valid_area = confirmed['AreaShape_Area_consensus'].notna()
    pos_area = ((confirmed['AreaShape_Area_consensus'] > 1) & valid_area).sum()
    neg_area = ((confirmed['AreaShape_Area_consensus'] < -1) & valid_area).sum()
    nan_area = (~valid_area).sum()
    print(f"\n  Among confirmed hits:")
    print(f"    Expanded (Area > +1):  {pos_area}")
    print(f"    Cytotoxic compact (Area < -1):  {neg_area}")
    print(f"    Other phenotype (|Area| < 1):   {n_confirmed - pos_area - neg_area - nan_area}")
    if nan_area > 0:
        print(f"    NaN AreaShape_Area:             {nan_area} (PROBLEM)")

out_hits = PROC_DIR / f'confirmed_hits_{plate}.csv'
hits_df.to_csv(out_hits, index=False)
print(f"\n  Saved: {out_hits}")

print(f"\n  Top 30 (by activity) with reproducibility info:")
print(hits_df.head(30).to_string(index=False))

# Step 8: Plot
print(f"\n{'-'*70}\nStep 8: Visualizations\n{'-'*70}")
from matplotlib.lines import Line2D

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
status_colors = {'inactive': '#BDC3C7', 'reproducible_only': '#F39C12',
                 'active_only': '#3498DB', 'confirmed': '#E74C3C'}
for _, row in hits_df.iterrows():
    if row['is_confirmed_hit']:
        c, s = status_colors['confirmed'], 50
    elif row['is_active_top5pct']:
        c, s = status_colors['active_only'], 25
    elif row['is_reproducible']:
        c, s = status_colors['reproducible_only'], 25
    else:
        c, s = status_colors['inactive'], 10
    rep_corr = row['Metadata_Replicate_corr']
    if not np.isnan(rep_corr):
        ax.scatter(row['Metadata_Activity'], rep_corr, c=c, s=s, alpha=0.7,
                   edgecolors='black', linewidths=0.3)
ax.axvline(final_threshold, color='red', linestyle='--', alpha=0.5)
ax.axhline(args.rep_corr_threshold, color='green', linestyle='--', alpha=0.5)
ax.set_xlabel('Activity score'); ax.set_ylabel('Replicate correlation')
ax.set_title('Robust hit calling')
legend = [Line2D([0],[0], marker='o', color='w', markerfacecolor=status_colors['confirmed'],
                 markersize=10, label=f'Confirmed (n={n_confirmed})')]
ax.legend(handles=legend, fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
if 'AreaShape_Area_consensus' in hits_df.columns:
    for _, row in hits_df.iterrows():
        area_val = row['AreaShape_Area_consensus']
        if pd.isna(area_val): continue
        if row['is_confirmed_hit']:
            if area_val > 1: c = '#E67E22'
            elif area_val < -1: c = '#9B59B6'
            else: c = '#E74C3C'
            s = 50
        else:
            c, s = '#BDC3C7', 12
        ax.scatter(area_val, row['Metadata_Activity'], c=c, s=s, alpha=0.7,
                   edgecolors='black', linewidths=0.3)
    ax.axhline(final_threshold, color='red', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('AreaShape_Area (consensus)'); ax.set_ylabel('Activity')
    ax.set_title('Hits: Activity vs Area')
    legend = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#E67E22',
               markersize=10, label='Expanded'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#9B59B6',
               markersize=10, label='Cytotoxic compact'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#E74C3C',
               markersize=10, label='Other'),
    ]
    ax.legend(handles=legend, fontsize=9); ax.grid(alpha=0.3)

plt.suptitle(f'Robust hit calling: {plate}', fontsize=12, fontweight='bold')
plt.tight_layout()
plot2 = OUT_DIR / f'{plate}_robust_activity.png'
plt.savefig(plot2, dpi=120, bbox_inches='tight')
print(f"  Robust activity plot saved: {plot2}")
plt.close()

# Step 9: Clustering
print(f"\n{'-'*70}\nStep 9: Clustering of confirmed hits\n{'-'*70}")
confirmed_hits = hits_df[hits_df['is_confirmed_hit']]['Metadata_Compound'].tolist()

if len(confirmed_hits) < 3:
    print(f"  Only {len(confirmed_hits)} confirmed hits — too few for clustering. Skipping.")
else:
    hit_profiles = consensus[consensus['Metadata_Compound'].isin(confirmed_hits)].copy()
    profile_data = hit_profiles[global_features].fillna(0).values
    print(f"  Clustering {len(confirmed_hits)} confirmed hits...")
    
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2)
        cluster_labels = clusterer.fit_predict(profile_data)
        method_used = 'HDBSCAN'
    except Exception as e:
        from sklearn.cluster import AgglomerativeClustering
        n_clusters = min(5, max(2, len(confirmed_hits)//4))
        cluster_labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(profile_data)
        method_used = f'Agglomerative (k={n_clusters})'
    
    hit_profiles['Cluster'] = cluster_labels
    n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = (cluster_labels == -1).sum()
    print(f"  Method: {method_used}")
    print(f"  Clusters: {n_clusters_found}, noise: {n_noise}")
    
    for cid in sorted(set(cluster_labels)):
        members = hit_profiles[hit_profiles['Cluster']==cid]
        if cid == -1:
            print(f"\n  Noise (unclustered): {len(members)} compounds")
        else:
            print(f"\n  Cluster {cid}: {len(members)} compounds")
        for _, m in members.iterrows():
            area = m.get('AreaShape_Area', np.nan)
            area_str = f"{area:+.2f}" if not pd.isna(area) else "NaN"
            print(f"    {m['Metadata_Compound']}: activity={m['Metadata_Activity']:.2f}, area={area_str}")
    
    cluster_out = PROC_DIR / f'hit_clusters_{plate}.csv'
    cluster_cols = ['Metadata_Compound', 'Metadata_Activity', 'Cluster']
    if 'AreaShape_Area' in hit_profiles.columns:
        cluster_cols.append('AreaShape_Area')
    hit_profiles[cluster_cols].to_csv(cluster_out, index=False)
    print(f"\n  Cluster assignments saved: {cluster_out}")

print(f"\n{'='*70}\n=== DONE ===\n{'='*70}")
print(f"Confirmed hits: {n_confirmed}")