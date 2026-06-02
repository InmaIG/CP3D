"""
01_merge_and_qc.py (v3 — fixes Metadata_Plate duplication)
==========================================================
Combina los CSVs de CellProfiler de una replica con el plate map procesado.
Genera CSV merged + QC plot con 4 paneles.

Compatible con:
- Plate maps tradicionales (1 placa fisica = 1 replica): C2386, C2387
- Plate maps multi-replica (1 placa fisica = N replicas tecnicas): C2388
  En este caso el plate map tiene columna Replicate y se filtra automaticamente.

CAMBIOS v3:
- Antes de mergear, elimina columnas del platemap que ya existen en spheroid
  (como Plate, que se anadia como Metadata_Plate y duplicaba).
- Limpia cualquier columna duplicada con sufijo _x/_y tras el merge.

Uso:
    python 01_merge_and_qc.py --plate C2386 --folder C2386R1
    python 01_merge_and_qc.py --plate C2388 --folder C2388R1
"""

import argparse
import sys
import re
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*tick_labels.*')

# ---------------- Paths ----------------
BASE = Path(r'C:\Users\Ianezi\Documents\CP3D\analysis')
RAW_DIR = BASE / 'data' / 'raw'
PLATEMAP_DIR = BASE / 'data' / 'platemaps' / 'processed'
PROC_DIR = BASE / 'data' / 'processed'
QC_DIR = BASE / 'results' / 'qc'

PROC_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Args ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--plate', required=True, help='Plate base name (e.g. C2386).')
parser.add_argument('--folder', required=True, help='Replicate folder name (e.g. C2386R1).')
args = parser.parse_args()

plate_base = args.plate
folder_name = args.folder

m = re.match(rf'{plate_base}(R\d+)', folder_name)
replicate_id = m.group(1) if m else None

print(f"\n{'='*70}\n=== MERGE + QC: {folder_name} ===\n{'='*70}")

# ---------------- Find CSVs ----------------
folder_path = RAW_DIR / folder_name
print(f"Looking in: {folder_path}")

def find_csv(folder, suffix):
    if not folder.exists():
        return None
    exact = folder / f'CP3D_{suffix}.csv'
    if exact.exists():
        return exact
    candidates = sorted(folder.glob(f'CP3D_*_{suffix}.csv'))
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        print(f"WARNING: Multiple matches; using {candidates[0].name}")
        return candidates[0]
    return None

sph_path = find_csv(folder_path, 'Spheroid_filter2')
img_path = find_csv(folder_path, 'Image')
platemap_path = PLATEMAP_DIR / f'platemap_{plate_base}_processed.csv'

print(f"Spheroid features: {sph_path}")
print(f"Image data:        {img_path}")
print(f"Plate map:         {platemap_path}")

if sph_path is None or not sph_path.exists():
    print(f"\nERROR: No se encuentra Spheroid CSV en {folder_path}")
    sys.exit(1)
if img_path is None or not img_path.exists():
    print(f"\nERROR: No se encuentra Image CSV en {folder_path}")
    sys.exit(1)
if not platemap_path.exists():
    print(f"\nERROR: No se encuentra plate map: {platemap_path}")
    sys.exit(1)

# ---------------- Load ----------------
sph = pd.read_csv(sph_path)
img = pd.read_csv(img_path)
platemap = pd.read_csv(platemap_path)

print(f"\nSpheroid CSV: {len(sph)} rows x {len(sph.columns)} cols")
print(f"Image CSV:    {len(img)} rows x {len(img.columns)} cols")
print(f"Plate map:    {len(platemap)} rows")

# ---------------- Filter plate map by replicate (if applicable) ----------------
if 'Replicate' in platemap.columns and replicate_id is not None:
    platemap_full = platemap.copy()
    platemap = platemap[platemap['Replicate'] == replicate_id].reset_index(drop=True)
    print(f"\nDetected multi-replicate platemap.")
    print(f"  Filtering to replicate {replicate_id}: {len(platemap)} wells "
          f"(of {len(platemap_full)} total in platemap)")

# Drop platemap-only metadata columns
cols_to_drop = [c for c in ['Source_Well', 'Replicate'] if c in platemap.columns]
if cols_to_drop:
    print(f"  Dropping platemap-only columns before merge: {cols_to_drop}")
    platemap = platemap.drop(columns=cols_to_drop)

# ---------------- Drop duplicate columns from spheroid ----------------
dup_cols = [c for c in sph.columns if c.endswith('.1')]
if dup_cols:
    print(f"\nDropping duplicate columns from spheroid: {dup_cols}")
    sph = sph.drop(columns=dup_cols)

# ---------------- Identify well column in spheroid CSV ----------------
well_col_candidates = ['Metadata_Well', 'Image_Metadata_Well', 'Well']
sph_well_col = None
for c in well_col_candidates:
    if c in sph.columns:
        sph_well_col = c
        break

if sph_well_col is None:
    print("\nERROR: No se encuentra columna de Well en spheroid CSV")
    sys.exit(1)

# ---------------- Add Metadata_Plate to sph if not already there ----------------
if 'Metadata_Plate' not in sph.columns:
    sph['Metadata_Plate'] = plate_base

# Normalize well naming
def normalize_well(w):
    m = re.match(r'^([A-Z]+)(\d+)$', str(w).strip())
    if not m:
        return str(w)
    return f"{m.group(1)}{int(m.group(2)):02d}"

sph[sph_well_col] = sph[sph_well_col].apply(normalize_well)
platemap['Well'] = platemap['Well'].apply(normalize_well)

# ---------------- IMPORTANT: drop columns from platemap that already exist in sph ----------------
# This prevents the "_x"/"_y" suffix duplication after merge.
sph_cols_set = set(sph.columns)

# Rename platemap columns: prefix with Metadata_ (except 'Well' which is the merge key)
platemap_renamed = platemap.copy()
platemap_renamed.columns = [
    f'Metadata_{c}' if not c.startswith('Metadata_') and c != 'Well' else c
    for c in platemap_renamed.columns
]

# Drop from platemap columns that already exist in sph (avoid duplicates)
overlap = set(platemap_renamed.columns) & sph_cols_set - {'Well'}
if overlap:
    print(f"\nColumns already in spheroid CSV (will be kept from sph, dropped from platemap):")
    for c in overlap:
        print(f"  {c}")
    platemap_renamed = platemap_renamed.drop(columns=list(overlap))

# ---------------- Merge ----------------
print(f"\nMerging features with plate map on Well...")
merged = sph.merge(
    platemap_renamed,
    left_on=sph_well_col,
    right_on='Well',
    how='left',
)

# Sanity: ensure no _x/_y suffix duplication
problem_cols = [c for c in merged.columns if c.endswith('_x') or c.endswith('_y')]
if problem_cols:
    print(f"\nWARNING: found duplicate-suffix columns after merge: {problem_cols}")
    print("  This should not happen. Cleaning up by keeping _x and dropping _y:")
    for c in problem_cols:
        if c.endswith('_x'):
            base = c[:-2]
            merged = merged.rename(columns={c: base})
        elif c.endswith('_y'):
            merged = merged.drop(columns=[c])

# Drop rows whose well is not in platemap
unmapped = merged['Metadata_Compound'].isna().sum()
print(f"  Merged: {len(merged)}")
print(f"  In features but not in plate map: {unmapped}")

if unmapped > 0:
    print(f"  These will be dropped (out-of-layout wells)")
    merged = merged[merged['Metadata_Compound'].notna()].reset_index(drop=True)

# Drop the redundant 'Well' column from merge if Metadata_Well also exists
if 'Well' in merged.columns and sph_well_col == 'Metadata_Well':
    merged = merged.drop(columns=['Well'])

# ---------------- QC: Well coverage ----------------
print(f"\n{'-'*70}\nQC: Well coverage\n{'-'*70}")

if 'Metadata_Plate' not in img.columns:
    img['Metadata_Plate'] = plate_base
img_well_col = sph_well_col if sph_well_col in img.columns else 'Metadata_Well'
if img_well_col in img.columns:
    img[img_well_col] = img[img_well_col].apply(normalize_well)

expected_wells = set(platemap['Well'])
img_wells = set(img[img_well_col]) if img_well_col in img.columns else set()
img_in_layout = img_wells & expected_wells

print(f"Wells imaged (in expected layout):  {len(img_in_layout)}")
if len(img_in_layout) > 0:
    print(f"Wells with valid spheroid:          {len(merged)} ({100*len(merged)/len(img_in_layout):.1f}%)")
    print(f"Wells without spheroid:             {len(img_in_layout) - len(merged)}")

# Compounds in failed wells
failed_wells = img_in_layout - set(merged[sph_well_col])
if len(failed_wells) > 0:
    failed_pm = platemap[platemap['Well'].isin(failed_wells)]
    print(f"\nCompounds in failed wells (breakdown):")
    print(f"  By type: {failed_pm['Well_type'].value_counts().to_dict()}")

# DMSO stats
dmso_in_platemap = platemap[platemap['Well_type']=='control']
dmso_valid = merged[merged['Metadata_Well_type']=='control']
print(f"\nDMSO wells:")
print(f"  Total expected: {len(dmso_in_platemap)}")
if len(dmso_in_platemap) > 0:
    print(f"  Valid:          {len(dmso_valid)} ({100*len(dmso_valid)/len(dmso_in_platemap):.1f}%)")
    print(f"  Failed:         {len(dmso_in_platemap) - len(dmso_valid)}")
    failed_dmso = sorted(set(dmso_in_platemap['Well']) - set(dmso_valid[sph_well_col]))
    if failed_dmso:
        print(f"    Wells: {failed_dmso[:30]}{'...' if len(failed_dmso)>30 else ''}")

# ---------------- QC: Feature sanity ----------------
print(f"\n{'-'*70}\nQC: Feature sanity checks\n{'-'*70}")

if 'AreaShape_Area' in merged.columns:
    print(f"\nSpheroid area by well type:")
    print(merged.groupby('Metadata_Well_type')['AreaShape_Area'].describe()[['count','mean','std','min','max']].to_string())
    
    dmso_med = dmso_valid['AreaShape_Area'].median() if len(dmso_valid) > 0 else None
    cpd_data = merged[merged['Metadata_Well_type']=='compound']
    cpd_med = cpd_data['AreaShape_Area'].median() if len(cpd_data) > 0 else None
    if dmso_med:
        print(f"\n  DMSO median area:      {dmso_med:.0f}")
    if cpd_med:
        print(f"  Compound median area:  {cpd_med:.0f}")
    if dmso_med and len(cpd_data) > 0:
        p5 = dmso_valid['AreaShape_Area'].quantile(0.05)
        toxic = (cpd_data['AreaShape_Area'] < p5).sum()
        print(f"  Compounds with area below DMSO P5 ({p5:.0f}): {toxic} ({100*toxic/len(cpd_data):.1f}% of compounds) - potential toxic")

# Feature count
n_meta = sum(1 for c in merged.columns if c.startswith('Metadata_') or c.startswith('FileName_') 
             or c.startswith('PathName_') or c in ('ImageNumber', 'ObjectNumber', 'Well'))
n_features = len(merged.columns) - n_meta
print(f"\nTotal feature columns: {n_features}")

# ---------------- Generate QC plot ----------------
print(f"\n{'-'*70}\nGenerating QC plot...\n{'-'*70}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Plate layout
ax = axes[0, 0]
ROWS = list('ABCDEFGHIJKLMNOP')
all_cols = sorted(set(int(w[1:]) for w in expected_wells)) if expected_wells else []

if all_cols:
    grid = np.full((len(ROWS), max(all_cols)), np.nan)
    for w in expected_wells:
        r = ROWS.index(w[0])
        c = int(w[1:]) - 1
        well_type = platemap[platemap['Well']==w]['Well_type'].values[0]
        if w in set(merged[sph_well_col]):
            if well_type == 'control':
                grid[r, c] = 2
            else:
                grid[r, c] = 1
        else:
            grid[r, c] = 0

    cmap = plt.cm.colors.ListedColormap(['#E74C3C', '#3498DB', '#27AE60'])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, aspect='auto')
    ax.set_xticks(range(0, max(all_cols), 2))
    ax.set_xticklabels([str(c+1) for c in range(0, max(all_cols), 2)], fontsize=8)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels(ROWS, fontsize=8)
    ax.set_title(f'Well status: {folder_name}', fontsize=11)
    ax.set_xlabel('Column')
    from matplotlib.patches import Patch
    legend = [Patch(color='#27AE60', label='Valid DMSO'),
              Patch(color='#3498DB', label='Valid compound'),
              Patch(color='#E74C3C', label='Failed')]
    ax.legend(handles=legend, loc='upper right', fontsize=8)

# Panel 2: Heatmap of areas
ax = axes[0, 1]
if 'AreaShape_Area' in merged.columns and all_cols:
    area_grid = np.full((len(ROWS), max(all_cols)), np.nan)
    for _, row in merged.iterrows():
        w = row[sph_well_col]
        r = ROWS.index(w[0])
        c = int(w[1:]) - 1
        area_grid[r, c] = row['AreaShape_Area']
    im = ax.imshow(area_grid, cmap='viridis', aspect='auto')
    ax.set_xticks(range(0, max(all_cols), 2))
    ax.set_xticklabels([str(c+1) for c in range(0, max(all_cols), 2)], fontsize=8)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels(ROWS, fontsize=8)
    ax.set_title(f'Spheroid area (px²)', fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046)

# Panel 3: Boxplot DMSO vs compounds
ax = axes[1, 0]
if 'AreaShape_Area' in merged.columns:
    data_to_plot = []
    labels = []
    if len(dmso_valid) > 0:
        data_to_plot.append(dmso_valid['AreaShape_Area'].dropna())
        labels.append(f'DMSO (n={len(dmso_valid)})')
    cpd_data = merged[merged['Metadata_Well_type']=='compound']
    if len(cpd_data) > 0:
        data_to_plot.append(cpd_data['AreaShape_Area'].dropna())
        labels.append(f'Compounds (n={len(cpd_data)})')
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#27AE60', '#3498DB']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax.set_ylabel('Spheroid Area (px²)')
    ax.set_title('Area: DMSO vs Compounds', fontsize=11)
    ax.grid(alpha=0.3, axis='y')

# Panel 4: Failed wells distribution
ax = axes[1, 1]
if failed_wells:
    failed_rows = [w[0] for w in failed_wells]
    row_counts = pd.Series(failed_rows).value_counts().sort_index()
    ax.bar(row_counts.index, row_counts.values, color='#E67E22', alpha=0.7)
    ax.set_ylabel('# Failed wells')
    ax.set_xlabel('Row')
    ax.set_title(f'Failed wells distribution (total={len(failed_wells)})')
    ax.grid(alpha=0.3, axis='y')

plt.suptitle(f'{folder_name} — QC summary', fontsize=12, fontweight='bold')
plt.tight_layout()

qc_plot_path = QC_DIR / f'{folder_name}_qc_plot.png'
plt.savefig(qc_plot_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"  QC plot saved: {qc_plot_path}")

# ---------------- Save merged data ----------------
out_merged = PROC_DIR / f'{folder_name}_merged.csv'
merged.to_csv(out_merged, index=False)
print(f"\n  Merged data saved: {out_merged}")
print(f"    Shape: {merged.shape}")

# ---------------- QC summary CSV ----------------
qc_summary = pd.DataFrame([{
    'folder': folder_name,
    'plate': plate_base,
    'replicate': replicate_id,
    'wells_imaged': len(img_in_layout),
    'wells_valid': len(merged),
    'success_rate_pct': round(100*len(merged)/len(img_in_layout), 1) if len(img_in_layout) > 0 else 0,
    'dmso_expected': len(dmso_in_platemap),
    'dmso_valid': len(dmso_valid),
    'compound_valid': (merged['Metadata_Well_type']=='compound').sum() if 'Metadata_Well_type' in merged.columns else 0,
    'area_median_dmso': float(dmso_med) if dmso_med else 0,
    'area_median_compound': float(cpd_med) if cpd_med else 0,
    'features_total': n_features,
}])
qc_csv_path = QC_DIR / f'{folder_name}_qc_summary.csv'
qc_summary.to_csv(qc_csv_path, index=False)
print(f"  QC summary saved: {qc_csv_path}")

print(f"\n{'='*70}\n=== DONE ===\n{'='*70}")
print(f"Next step: run 02_normalize.py --folder {folder_name}")