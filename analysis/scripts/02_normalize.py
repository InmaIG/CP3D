"""
02_normalize.py (v3)
====================
Normalize CellProfiler features against DMSO with proper handling of:
- Features with near-zero MAD in DMSO (would cause division explosions)
- Extreme outliers post-normalization (winsorization)

Designed for bioactive compound libraries where DMSO is the only valid
control reference (most compounds are expected to be active).

Inputs:
- data\\processed\\<folder>_merged.csv

Outputs:
- data\\processed\\<folder>_normalized_<normby>.csv
- data\\processed\\<folder>_normalized_<normby>_selected.csv
- results\\norm\\<folder>_norm_<normby>_qc.png
- results\\norm\\<folder>_norm_<normby>_position.png
- results\\norm\\<folder>_norm_<normby>_summary.txt

Uso:
    python 02_normalize.py --folder C2386R1
    python 02_normalize.py --folder C2386R1 --mad_min 0.005 --winsorize 10
"""

import argparse
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pycytominer import normalize, feature_select

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*tick_labels.*')

BASE = Path(r'C:\Users\Ianezi\Documents\CP3D\analysis')
PROC_DIR = BASE / 'data' / 'processed'
NORM_DIR = BASE / 'results' / 'norm'
NORM_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--folder', required=True)
parser.add_argument('--normalize_by', default='dmso',
                    choices=['dmso', 'all', 'compounds'])
parser.add_argument('--mad_min', type=float, default=0.01,
                    help='Minimum DMSO MAD for a feature to be kept (default: 0.01).'
                         ' Features with smaller MAD are unstable and get dropped.')
parser.add_argument('--winsorize', type=float, default=10.0,
                    help='Clip normalized values to +/- this value (default: 10).')
parser.add_argument('--corr_threshold', type=float, default=0.9)
parser.add_argument('--variance_threshold', type=float, default=0.01)
args = parser.parse_args()

folder = args.folder
norm_by = args.normalize_by

in_path = PROC_DIR / f'{folder}_merged.csv'
print(f"\n{'='*70}\n=== NORMALIZE: {folder}  (control: {norm_by}) ===\n{'='*70}")
print(f"Input: {in_path}")
print(f"MAD floor: {args.mad_min}  Winsorize: +/-{args.winsorize}")

if not in_path.exists():
    print(f"\nERROR: No se encuentra {in_path}")
    sys.exit(1)

df = pd.read_csv(in_path)
print(f"\nLoaded: {len(df)} rows x {len(df.columns)} cols")

meta_cols = [c for c in df.columns if c.startswith('Metadata_') 
             or c.startswith('FileName_') or c.startswith('PathName_')
             or c in ['ImageNumber', 'ObjectNumber']]
feature_cols = [c for c in df.columns if c not in meta_cols]
print(f"  Metadata: {len(meta_cols)}  Features: {len(feature_cols)}")

if 'Metadata_Compound' not in df.columns:
    print("\nERROR: Metadata_Compound missing.")
    sys.exit(1)

n_dmso = (df['Metadata_Compound'] == 'DMSO').sum()
n_compound = (df['Metadata_Well_type'] == 'compound').sum()
print(f"\nWell counts: DMSO={n_dmso}, Compounds={n_compound}")

# Drop rows with all-NaN features
features_only = df[feature_cols]
all_nan_rows = features_only.isna().all(axis=1)
if all_nan_rows.sum() > 0:
    df = df[~all_nan_rows].reset_index(drop=True)

# ---------------- Pre-filter: features with reliable DMSO MAD ----------------
# This is the KEY new step: drop features that have near-zero MAD in DMSO,
# because dividing by ~0 generates explosions in the normalized values.
print(f"\n{'-'*70}\nStep 0: Pre-filter features by DMSO MAD\n{'-'*70}")

dmso_df = df[df['Metadata_Compound'] == 'DMSO']
print(f"  DMSO subset: {len(dmso_df)} wells")

mad_per_feature = {}
for f in feature_cols:
    vals = dmso_df[f].dropna()
    if len(vals) < 5:
        mad_per_feature[f] = 0
        continue
    med = vals.median()
    mad = (vals - med).abs().median()
    mad_per_feature[f] = mad

mad_series = pd.Series(mad_per_feature)
print(f"\n  DMSO MAD distribution across features:")
print(f"    min:    {mad_series.min():.6f}")
print(f"    p5:     {mad_series.quantile(0.05):.6f}")
print(f"    median: {mad_series.median():.6f}")
print(f"    p95:    {mad_series.quantile(0.95):.6f}")
print(f"    max:    {mad_series.max():.6f}")

unstable = mad_series[mad_series < args.mad_min].index.tolist()
print(f"\n  Features with MAD < {args.mad_min}: {len(unstable)} (will be dropped)")
if 0 < len(unstable) <= 20:
    for f in unstable[:20]:
        print(f"    {f}: MAD={mad_series[f]:.6f}")

stable_features = [f for f in feature_cols if f not in unstable]
print(f"\n  Features kept: {len(stable_features)} (of {len(feature_cols)})")

if len(stable_features) < 50:
    print(f"\nWARNING: Solo quedan {len(stable_features)} features. Considera bajar mad_min.")

# Drop unstable features from df
df_stable = df.drop(columns=unstable)
feature_cols = stable_features

# ---------------- Choose reference ----------------
if norm_by == 'dmso':
    sample_query = "Metadata_Compound == 'DMSO'"
    n_ref = n_dmso
    print(f"\n  Reference: DMSO ({n_ref} wells)")
elif norm_by == 'compounds':
    sample_query = "Metadata_Well_type == 'compound'"
    n_ref = n_compound
    print(f"\n  Reference: compound wells ({n_ref})")
elif norm_by == 'all':
    df_stable['_ref_marker'] = 1
    sample_query = "_ref_marker == 1"
    n_ref = len(df_stable)
    print(f"\n  Reference: all wells ({n_ref})")

# ---------------- Normalize ----------------
print(f"\n{'-'*70}\nStep 1: Normalize (mad_robustize)\n{'-'*70}")

normalized = normalize(
    profiles=df_stable,
    features=feature_cols,
    meta_features=meta_cols,
    samples=sample_query,
    method='mad_robustize',
)

if '_ref_marker' in normalized.columns:
    normalized = normalized.drop(columns=['_ref_marker'])
if '_ref_marker' in df_stable.columns:
    df_stable = df_stable.drop(columns=['_ref_marker'])

print(f"  Normalized shape: {normalized.shape}")

# ---------------- Winsorize ----------------
print(f"\n{'-'*70}\nStep 2: Winsorize at +/-{args.winsorize}\n{'-'*70}")

# Count extreme values pre-winsorize
extreme_before = (normalized[feature_cols].abs() > args.winsorize).sum().sum()
total_cells = len(normalized) * len(feature_cols)
print(f"  Values |x| > {args.winsorize} before winsorize: {extreme_before} "
      f"({100*extreme_before/total_cells:.4f}% of cells)")

normalized[feature_cols] = normalized[feature_cols].clip(
    lower=-args.winsorize, upper=args.winsorize
)

# Sanity check
print(f"\n  Sanity check post-winsorize on reference:")
if norm_by == 'dmso':
    ref_norm = normalized[normalized['Metadata_Compound']=='DMSO']
elif norm_by == 'compounds':
    ref_norm = normalized[normalized['Metadata_Well_type']=='compound']
else:
    ref_norm = normalized
for f in feature_cols[:5]:
    if f in ref_norm.columns:
        med = ref_norm[f].median()
        mad = (ref_norm[f] - ref_norm[f].median()).abs().median()
        print(f"    {f[:55]:55s} median={med:+.4f} MAD={mad:.4f}")

# Show DMSO position if normalized by something else
if norm_by != 'dmso':
    print(f"\n  DMSO position after normalizing against {norm_by}:")
    dmso_n = normalized[normalized['Metadata_Compound']=='DMSO']
    for f in feature_cols[:5]:
        if f in dmso_n.columns:
            print(f"    {f[:55]:55s} DMSO median={dmso_n[f].median():+.4f}")

out_norm_path = PROC_DIR / f'{folder}_normalized_{norm_by}.csv'
normalized.to_csv(out_norm_path, index=False)
print(f"\n  Saved: {out_norm_path}")

# ---------------- Feature selection ----------------
print(f"\n{'-'*70}\nStep 3: Feature selection\n{'-'*70}")

n_features_before = len(feature_cols)
selected = feature_select(
    profiles=normalized,
    features=feature_cols,
    operation=['variance_threshold', 'correlation_threshold',
               'drop_na_columns', 'blocklist'],
    na_cutoff=0.05,
    corr_threshold=args.corr_threshold,
    freq_cut=args.variance_threshold,
    unique_cut=0.1,
)

selected_features = [c for c in selected.columns if c in feature_cols]
n_features_after = len(selected_features)
print(f"  Before: {n_features_before}")
print(f"  After:  {n_features_after}")
print(f"  Removed: {n_features_before - n_features_after} "
      f"({100*(n_features_before-n_features_after)/n_features_before:.1f}%)")

out_sel_path = PROC_DIR / f'{folder}_normalized_{norm_by}_selected.csv'
selected.to_csv(out_sel_path, index=False)
print(f"\n  Saved: {out_sel_path}")

# ---------------- QC plot 1: before/after ----------------
print(f"\n{'-'*70}\nGenerating QC plots...\n{'-'*70}")

qc_features = []
for pat in ['AreaShape_Area', 'Intensity_MeanIntensity_DNA',
            'Intensity_MeanIntensity_MITO', 'Intensity_MeanIntensity_Mito']:
    for f in feature_cols:
        if pat in f:
            qc_features.append(f)
            break
# Add a Texture feature if available
for f in feature_cols:
    if 'Texture' in f and 'AGP' in f:
        qc_features.append(f)
        break

# Dedup, max 4
qc_features = list(dict.fromkeys(qc_features))[:4]
print(f"  QC features: {qc_features}")

if qc_features:
    fig, axes = plt.subplots(2, len(qc_features), figsize=(5*len(qc_features), 9))
    if len(qc_features) == 1:
        axes = axes.reshape(2, 1)
    
    for col_idx, feat in enumerate(qc_features):
        ax = axes[0, col_idx]
        raw_dmso = df_stable[df_stable['Metadata_Compound']=='DMSO'][feat].dropna()
        raw_cpd = df_stable[df_stable['Metadata_Well_type']=='compound'][feat].dropna()
        bp = ax.boxplot([raw_dmso, raw_cpd], tick_labels=['DMSO', 'Cpd'],
                        patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#2ECC71', '#3498DB']):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        ax.set_title(f'RAW\n{feat[:50]}', fontsize=9)
        ax.set_ylabel('Raw value')
        
        ax = axes[1, col_idx]
        norm_dmso = normalized[normalized['Metadata_Compound']=='DMSO'][feat].dropna()
        norm_cpd = normalized[normalized['Metadata_Well_type']=='compound'][feat].dropna()
        bp = ax.boxplot([norm_dmso, norm_cpd], tick_labels=['DMSO', 'Cpd'],
                        patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#2ECC71', '#3498DB']):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_title(f'NORMALIZED (vs {norm_by})\n{feat[:50]}', fontsize=9)
        ax.set_ylabel('Normalized')
    
    plt.suptitle(f'Normalization QC: {folder} (control={norm_by})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    qc_plot = NORM_DIR / f'{folder}_norm_{norm_by}_qc.png'
    plt.savefig(qc_plot, dpi=120, bbox_inches='tight')
    print(f"  QC plot saved: {qc_plot}")
    plt.close()

# ---------------- QC plot 2: position effect ----------------
ROWS = list('ABCDEFGHIJKLMNOP')
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Use SELECTED features for activity (cleaner signal)
sel_only = normalized[selected_features].fillna(0)
activity = np.sqrt((sel_only ** 2).sum(axis=1)) / np.sqrt(len(selected_features))
# normalized by sqrt(n_features) so values are in similar scale to feature values

normalized['_activity'] = activity

ax = axes[0]
grid = np.full((16, 24), np.nan)
for _, row in normalized.iterrows():
    w = row['Metadata_Well']
    r = ROWS.index(w[0])
    c = int(w[1:]) - 1
    grid[r, c] = row['_activity']
im = ax.imshow(grid, cmap='magma', aspect='equal')
ax.set_xticks(range(24)); ax.set_xticklabels(range(1, 25), fontsize=8)
ax.set_yticks(range(16)); ax.set_yticklabels(ROWS, fontsize=8)
ax.set_title(f'Activity heatmap (RMS profile)\ncontrol={norm_by}', fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[1]
for col_num in range(1, 25):
    col_data = normalized[normalized['Metadata_Col']==col_num]['_activity'].dropna()
    if len(col_data) > 0:
        ax.scatter([col_num]*len(col_data), col_data, alpha=0.5, s=15, c='#3498DB')
        ax.scatter([col_num], [col_data.median()], s=80, c='red', marker='_', linewidth=3)
ax.set_xlabel('Column'); ax.set_ylabel('Activity (RMS profile)')
ax.set_title(f'Activity by column (red = median)', fontsize=11)
ax.set_xticks(range(1, 25))
ax.grid(True, alpha=0.3)

plt.suptitle(f'Position effect check: {folder} (control={norm_by})',
             fontsize=12, fontweight='bold')
plt.tight_layout()
pos_plot = NORM_DIR / f'{folder}_norm_{norm_by}_position.png'
plt.savefig(pos_plot, dpi=120, bbox_inches='tight')
print(f"  Position plot saved: {pos_plot}")
plt.close()
normalized = normalized.drop(columns=['_activity'])

# ---------------- Summary ----------------
summary_path = NORM_DIR / f'{folder}_norm_{norm_by}_summary.txt'
with open(summary_path, 'w') as f:
    f.write(f"Normalization summary: {folder}\n")
    f.write(f"="*70 + "\n\n")
    f.write(f"Input: {in_path.name}\n")
    f.write(f"Reference: {norm_by} (n={n_ref})\n")
    f.write(f"MAD floor: {args.mad_min}\n")
    f.write(f"Winsorize: +/-{args.winsorize}\n\n")
    f.write(f"Total rows: {len(normalized)}\n")
    f.write(f"  DMSO: {(normalized['Metadata_Compound']=='DMSO').sum()}\n")
    f.write(f"  Compounds: {(normalized['Metadata_Well_type']=='compound').sum()}\n\n")
    f.write(f"Features:\n")
    f.write(f"  Original: {len(stable_features) + len(unstable)}\n")
    f.write(f"  After MAD filter: {len(stable_features)} ")
    f.write(f"(dropped {len(unstable)} with MAD < {args.mad_min})\n")
    f.write(f"  After feature selection: {n_features_after}\n\n")
    f.write(f"Outputs:\n")
    f.write(f"  Normalized: {out_norm_path.name}\n")
    f.write(f"  Selected: {out_sel_path.name}\n")
print(f"  Summary saved: {summary_path}")

print(f"\n{'='*70}\n=== DONE ===\n{'='*70}")
print(f"Final file: {out_sel_path}")