"""
split_C2388_replicates.py
=========================
Adapta C2388 al pipeline estandar dividiendo la unica placa fisica
en 4 carpetas virtuales (una por replica tecnica), de modo que los
scripts 01-04 funcionen sin cambios.

Operacion:
  Input:  data\\raw\\C2388\\CP3D_C2388_*_Image.csv y *_Spheroid_filter2.csv
  Output: data\\raw\\C2388R1\\, C2388R2\\, C2388R3\\, C2388R4\\
          (cada una con su propio Image.csv y Spheroid_filter2.csv)

La asignacion a replica se hace usando platemap_C2388_processed.csv
que tiene la columna Replicate.

Uso:
    python split_C2388_replicates.py
    python split_C2388_replicates.py --source_folder C2388 --plate C2388
"""

import argparse
import sys
import re
from pathlib import Path
import pandas as pd

# ---------------- Paths ----------------
BASE = Path(r'C:\Users\Ianezi\Documents\CP3D\analysis')
RAW_DIR = BASE / 'data' / 'raw'
PLATEMAP_DIR = BASE / 'data' / 'platemaps' / 'processed'

# ---------------- Args ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--source_folder', default='C2388',
                    help='Carpeta con los CSVs originales de CellProfiler (default: C2388).')
parser.add_argument('--plate', default='C2388',
                    help='Nombre base de la placa (default: C2388).')
args = parser.parse_args()

source_folder = RAW_DIR / args.source_folder
plate = args.plate
platemap_path = PLATEMAP_DIR / f'platemap_{plate}_processed.csv'

print(f"\n{'='*70}\n=== SPLIT C2388 INTO REPLICATE FOLDERS ===\n{'='*70}")
print(f"Source folder: {source_folder}")
print(f"Plate map:     {platemap_path}")

# ---------------- Find CSVs ----------------
def find_csv(folder, suffix):
    """Find a CSV in folder matching CP3D_*_<suffix>.csv or CP3D_<suffix>.csv."""
    if not folder.exists():
        return None
    # Try exact match first
    exact = folder / f'CP3D_{suffix}.csv'
    if exact.exists():
        return exact
    # Try wildcard
    candidates = sorted(folder.glob(f'CP3D_*_{suffix}.csv'))
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        print(f"WARNING: Multiple matches for *_{suffix}.csv:")
        for c in candidates:
            print(f"  {c}")
        return candidates[0]
    return None


sph_path = find_csv(source_folder, 'Spheroid_filter2')
img_path = find_csv(source_folder, 'Image')

if sph_path is None:
    print(f"\nERROR: No se encuentra ningun *_Spheroid_filter2.csv en {source_folder}")
    sys.exit(1)
if img_path is None:
    print(f"\nERROR: No se encuentra ningun *_Image.csv en {source_folder}")
    sys.exit(1)
if not platemap_path.exists():
    print(f"\nERROR: No se encuentra plate map procesado: {platemap_path}")
    print("       Ejecuta process_platemap_C2388.py primero.")
    sys.exit(1)

print(f"\nFiles found:")
print(f"  Spheroid CSV: {sph_path.name}")
print(f"  Image CSV:    {img_path.name}")

# ---------------- Load ----------------
print(f"\n{'-'*70}\nLoad files\n{'-'*70}")

sph = pd.read_csv(sph_path)
img = pd.read_csv(img_path)
platemap = pd.read_csv(platemap_path)

print(f"  Spheroid CSV: {len(sph)} rows x {len(sph.columns)} cols")
print(f"  Image CSV:    {len(img)} rows x {len(img.columns)} cols")
print(f"  Plate map:    {len(platemap)} wells")

if 'Replicate' not in platemap.columns:
    print(f"\nERROR: El plate map no tiene columna 'Replicate'.")
    print("       Asegurate de regenerarlo con process_platemap_C2388.py.")
    sys.exit(1)

# ---------------- Build well -> replicate map ----------------
well_to_rep = dict(zip(platemap['Well'], platemap['Replicate']))

# Sanity check unique values
unique_reps = platemap['Replicate'].unique()
print(f"\n  Replicates in platemap: {sorted(unique_reps)}")
for rep in sorted(unique_reps):
    sub = platemap[platemap['Replicate']==rep]
    print(f"    {rep}: {len(sub)} wells "
          f"({(sub['Well_type']=='compound').sum()} compounds + "
          f"{(sub['Well_type']=='control').sum()} DMSO)")

# ---------------- Identify the well column in the spheroid CSV ----------------
# CellProfiler usually outputs Metadata_Well or similar
well_col_candidates = ['Metadata_Well', 'Image_Metadata_Well', 'Well']
sph_well_col = None
for c in well_col_candidates:
    if c in sph.columns:
        sph_well_col = c
        break

if sph_well_col is None:
    print(f"\nERROR: No se encuentra columna de Well en spheroid CSV.")
    print(f"       Columnas disponibles: {[c for c in sph.columns if 'Well' in c or 'well' in c]}")
    sys.exit(1)

img_well_col = None
for c in well_col_candidates:
    if c in img.columns:
        img_well_col = c
        break

print(f"\n  Spheroid well column: {sph_well_col}")
print(f"  Image well column:    {img_well_col}")

# ---------------- Assign replicate ----------------
print(f"\n{'-'*70}\nAssign replicate to each row\n{'-'*70}")

# Normalize well naming if necessary (A1 vs A01)
def normalize_well(w):
    m = re.match(r'^([A-Z]+)(\d+)$', str(w).strip())
    if not m:
        return str(w)
    return f"{m.group(1)}{int(m.group(2)):02d}"

sph['_well_norm'] = sph[sph_well_col].apply(normalize_well)
img['_well_norm'] = img[img_well_col].apply(normalize_well)

sph['_replicate'] = sph['_well_norm'].map(well_to_rep)
img['_replicate'] = img['_well_norm'].map(well_to_rep)

# Stats
n_sph_assigned = sph['_replicate'].notna().sum()
n_sph_unassigned = sph['_replicate'].isna().sum()
print(f"\n  Spheroid rows:")
print(f"    Assigned to a replicate: {n_sph_assigned}")
print(f"    Not in any replicate (cols 21-24 empty): {n_sph_unassigned}")

if n_sph_unassigned > 0:
    unassigned_wells = sph[sph['_replicate'].isna()]['_well_norm'].unique()
    print(f"    Unassigned wells: {sorted(unassigned_wells)[:10]}")

# Distribution per replicate
print(f"\n  Distribution by replicate:")
for rep in ['R1', 'R2', 'R3', 'R4']:
    n_sph = (sph['_replicate']==rep).sum()
    n_img = (img['_replicate']==rep).sum()
    print(f"    {rep}: {n_sph} spheroids, {n_img} images")

# ---------------- Split and save ----------------
print(f"\n{'-'*70}\nSplit and save\n{'-'*70}")

# Get the prefix of the original CSV name (e.g. CP3D_C2388_R1 or CP3D_C2388)
sph_stem = sph_path.stem  # CP3D_C2388_R1_Spheroid_filter2 or similar
img_stem = img_path.stem

# Prefix without _Spheroid_filter2 suffix
sph_prefix = sph_stem.replace('_Spheroid_filter2', '')
img_prefix = img_stem.replace('_Image', '')

for rep in ['R1', 'R2', 'R3', 'R4']:
    out_folder = RAW_DIR / f'{plate}{rep}'
    out_folder.mkdir(parents=True, exist_ok=True)
    
    sph_sub = sph[sph['_replicate']==rep].drop(columns=['_well_norm', '_replicate']).copy()
    img_sub = img[img['_replicate']==rep].drop(columns=['_well_norm', '_replicate']).copy()
    
    # Use a consistent naming: CP3D_<plate>_<rep>_Spheroid_filter2.csv
    sph_out = out_folder / f'CP3D_{plate}_{rep}_Spheroid_filter2.csv'
    img_out = out_folder / f'CP3D_{plate}_{rep}_Image.csv'
    
    sph_sub.to_csv(sph_out, index=False)
    img_sub.to_csv(img_out, index=False)
    
    print(f"  {rep}: {len(sph_sub)} spheroids, {len(img_sub)} images")
    print(f"    -> {sph_out.name}")
    print(f"    -> {img_out.name}")

print(f"\n{'='*70}\n=== DONE ===\n{'='*70}")
print(f"\nNow you can run the standard pipeline for each replicate:")
print(f"  python 01_merge_and_qc.py --plate {plate} --folder {plate}R1")
print(f"  python 02_normalize.py --folder {plate}R1")
print(f"  ... (R2, R3, R4)")
print(f"\nThen multi-replicate analysis:")
print(f"  python 03_combine_and_score.py --plate {plate} \\")
print(f"      --replicates {plate}R1 {plate}R2 {plate}R3 {plate}R4")
print(f"  python 04_robust_analysis.py --plate {plate} \\")
print(f"      --replicates {plate}R1 {plate}R2 {plate}R3 {plate}R4")