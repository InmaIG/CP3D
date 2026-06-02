"""
05_merge_hits_chemistry.py (v3 — adds min_replicates filter for confidence)
==========================================================================
Cruza confirmed hits con info quimica + biologica:
- Drug name, targets, genes, MoA, n_targets (promiscuidad)
- Jerarquias ChEMBL/GTOPDB

NUEVO en v3:
- Filtro --min_replicates (default 3): excluye hits con N_replicates < min
- Genera 2 outputs:
    * <plate>_hits_with_chemistry.csv         (filtrados, robustos, para downstream)
    * <plate>_hits_excluded_low_reps.csv      (descartados, para auditoria/informe)

Uso:
    python 05_merge_hits_chemistry.py --plate C2386
    python 05_merge_hits_chemistry.py --plate C2386 --min_replicates 3
    python 05_merge_hits_chemistry.py --plate C2386 --min_replicates 2
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

BASE = Path(r'C:\Users\Ianezi\Documents\CP3D\analysis')
PROC_DIR = BASE / 'data' / 'processed'
HITS_DIR = BASE / 'results' / 'hits_summary'
HITS_DIR.mkdir(parents=True, exist_ok=True)

CHEMISTRY_FILE = Path(r'C:\Users\Ianezi\Documents\CP3D\EOS_compounds_MoA.csv')

parser = argparse.ArgumentParser()
parser.add_argument('--plate', required=True)
parser.add_argument('--min_replicates', type=int, default=3,
                    help='Minimum N_replicates required to keep a hit (default: 3)')
args = parser.parse_args()
plate = args.plate
min_reps = args.min_replicates

print(f"\n{'='*70}\n=== MERGE HITS + CHEMISTRY/MoA: {plate} ===\n{'='*70}")
print(f"Minimum replicates required: {min_reps}")

hits_path = PROC_DIR / f'confirmed_hits_{plate}.csv'
print(f"Hits file:      {hits_path}")
print(f"Chemistry file: {CHEMISTRY_FILE}")

if not hits_path.exists():
    print(f"\nERROR: No se encuentra {hits_path}")
    sys.exit(1)
if not CHEMISTRY_FILE.exists():
    print(f"\nERROR: No se encuentra {CHEMISTRY_FILE}")
    sys.exit(1)

hits = pd.read_csv(hits_path)
try:
    chem = pd.read_csv(CHEMISTRY_FILE, encoding='latin-1')
except UnicodeDecodeError:
    chem = pd.read_csv(CHEMISTRY_FILE, encoding='utf-8', errors='replace')

print(f"\nHits dataframe:      {hits.shape[0]} rows x {hits.shape[1]} cols")
print(f"Chemistry dataframe: {chem.shape[0]} rows x {chem.shape[1]} cols")

# Identify EOS_id column
eos_col = None
for cand in ['EOS', 'EOS_id', 'eos_id']:
    if cand in chem.columns:
        eos_col = cand
        break
if eos_col is None:
    print("\nERROR: No se encuentra columna EOS")
    sys.exit(1)
print(f"\nUsing EOS column: {eos_col}")

# Categorize hits
print(f"\n{'-'*70}\nCategorize hits\n{'-'*70}")
def categorize(row):
    if row['is_confirmed_hit']:
        return 'confirmed_hit'
    if row['is_active_top5pct'] and not row['is_reproducible']:
        if row['Metadata_n_replicates'] == 1:
            return 'highly_toxic_no_replicates'
        return 'active_only'
    if row['is_reproducible'] and not row['is_active_top5pct']:
        return 'reproducible_only'
    return 'inactive'

hits['Hit_category'] = hits.apply(categorize, axis=1)
print(f"\nHit category distribution (full):")
print(hits['Hit_category'].value_counts().to_string())

def phenotype(row):
    if pd.isna(row['AreaShape_Area_consensus']):
        return 'unknown'
    a = row['AreaShape_Area_consensus']
    if a > 1: return 'expanded'
    if a < -1: return 'shrunken'
    return 'stable'

hits['Phenotype'] = hits.apply(phenotype, axis=1)

# ============================================================
# Initial filter: keep candidate hits (confirmed + highly_toxic)
# ============================================================
print(f"\n{'-'*70}\nFilter candidate hits\n{'-'*70}")
to_keep = ['confirmed_hit', 'highly_toxic_no_replicates']
hits_candidates = hits[hits['Hit_category'].isin(to_keep)].copy()
print(f"  Initial candidate hits: {len(hits_candidates)}")
print(f"    confirmed_hit:                {(hits_candidates['Hit_category']=='confirmed_hit').sum()}")
print(f"    highly_toxic_no_replicates:  {(hits_candidates['Hit_category']=='highly_toxic_no_replicates').sum()}")

if len(hits_candidates) == 0:
    print("\n  No candidate hits.")
    sys.exit(0)

# ============================================================
# NEW: Apply min_replicates filter
# ============================================================
print(f"\n{'-'*70}\nApply replicate filter (N_replicates >= {min_reps})\n{'-'*70}")
robust_mask = hits_candidates['Metadata_n_replicates'] >= min_reps
hits_robust = hits_candidates[robust_mask].copy()
hits_excluded = hits_candidates[~robust_mask].copy()

print(f"  Hits passing filter (N>={min_reps}):    {len(hits_robust)}")
print(f"  Hits excluded (N<{min_reps}):           {len(hits_excluded)}")

if len(hits_excluded) > 0:
    print(f"\n  Excluded hits (insufficient replicates):")
    excl_summary = hits_excluded[['Metadata_Compound','Hit_category',
                                    'Metadata_n_replicates','Metadata_Activity']].copy()
    excl_summary.columns = ['EOS_id','Hit_category','N_replicates','Activity_score']
    print(excl_summary.to_string(index=False))

# Merge BOTH (robust + excluded) with chemistry — for both output files
def merge_with_chem(df, label):
    if len(df) == 0:
        return pd.DataFrame()
    merged = df.merge(chem, left_on='Metadata_Compound', right_on=eos_col, how='left')
    n_with = merged['EUopen_name'].notna().sum()
    n_without = merged['EUopen_name'].isna().sum()
    print(f"  [{label}] With chemistry info: {n_with}, without: {n_without}")
    if n_without > 0:
        missing = merged[merged['EUopen_name'].isna()]['Metadata_Compound'].tolist()
        print(f"    Missing EOS_ids: {missing[:10]}")
    return merged

print(f"\n{'-'*70}\nMerge with chemistry/MoA\n{'-'*70}")
merged_robust = merge_with_chem(hits_robust, 'ROBUST')
merged_excluded = merge_with_chem(hits_excluded, 'EXCLUDED')

# ============================================================
# Build output table — for both robust and excluded
# ============================================================
final_cols = {
    'Metadata_Rank': 'Rank',
    'Metadata_Compound': 'EOS_id',
    'Hit_category': 'Hit_category',
    'Phenotype': 'Phenotype',
    'Metadata_Activity': 'Activity_score',
    'Metadata_Replicate_corr': 'Replicate_correlation',
    'Metadata_n_replicates': 'N_replicates',
    'Metadata_Replicates_used': 'Replicates_used',
    'AreaShape_Area_consensus': 'AreaShape_Area_zscore',
    'EUopen_name': 'Drug_name',
    'EUopen_target_name': 'Target_name',
    'EUopen_gene_name': 'Gene_name',
    'EUopen_moa': 'MoA',
    'EUopen_target_type': 'Target_type',
    'EUopen_no. targets': 'N_targets',
    'EUopen_smiles': 'SMILES',
    'EUopen_inchikey': 'InChIKey',
    'EUopen_mw': 'Molecular_weight',
    'EUopen_cas': 'CAS',
    'EUopen_synonyms': 'Synonyms',
    'EUopen_GTOPDB [LEVEL 1]': 'GTOPDB_L1',
    'EUopen_GTOPDB [LEVEL 2]': 'GTOPDB_L2',
    'EUopen_GTOPDB [LEVEL 3]': 'GTOPDB_L3',
    'EUopen_ChEMBL [LEVEL 1]': 'ChEMBL_L1',
    'EUopen_ChEMBL [LEVEL 2]': 'ChEMBL_L2',
    'EUopen_ChEMBL [LEVEL 3]': 'ChEMBL_L3',
    'EUopen_Reactome [LEVEL 1]': 'Reactome_L1',
}

def build_final(merged_df):
    if len(merged_df) == 0:
        return pd.DataFrame()
    available = {k: v for k, v in final_cols.items() if k in merged_df.columns}
    out = merged_df[list(available.keys())].rename(columns=available)
    out = out.sort_values('Rank').reset_index(drop=True)
    return out

print(f"\n{'-'*70}\nBuild final tables\n{'-'*70}")
output_robust = build_final(merged_robust)
output_excluded = build_final(merged_excluded)

# Save robust hits (the ones used downstream)
out_path = HITS_DIR / f'{plate}_hits_with_chemistry.csv'
output_robust.to_csv(out_path, index=False)
print(f"  Saved (robust):    {out_path}    [{output_robust.shape}]")

# Save excluded hits (for transparency/auditability)
if len(output_excluded) > 0:
    out_excl_path = HITS_DIR / f'{plate}_hits_excluded_low_reps.csv'
    output_excluded['Exclusion_reason'] = f'N_replicates < {min_reps}'
    output_excluded.to_csv(out_excl_path, index=False)
    print(f"  Saved (excluded):  {out_excl_path}    [{output_excluded.shape}]")
else:
    print(f"  No excluded hits (all candidates had N>={min_reps})")

# ============================================================
# Display robust hits (downstream-ready)
# ============================================================
print(f"\n{'-'*70}\nROBUST hits — downstream-ready (concise view)\n{'-'*70}")
if len(output_robust) > 0:
    display_cols = ['Rank', 'EOS_id', 'Drug_name', 'Phenotype', 'Activity_score',
                    'Replicate_correlation', 'N_replicates', 'AreaShape_Area_zscore',
                    'Target_name', 'MoA', 'N_targets']
    display_cols = [c for c in display_cols if c in output_robust.columns]
    display = output_robust[display_cols].copy()
    for c in ['Target_name']:
        if c in display.columns:
            display[c] = display[c].fillna('').astype(str).apply(
                lambda s: s if len(s) < 50 else s[:47]+'...')
    for c in ['Activity_score', 'Replicate_correlation', 'AreaShape_Area_zscore']:
        if c in display.columns:
            display[c] = display[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else 'NaN')
    print(display.to_string(index=False, max_colwidth=50))

# ============================================================
# Display excluded hits (for transparency)
# ============================================================
if len(output_excluded) > 0:
    print(f"\n{'-'*70}\nEXCLUDED hits (N_replicates < {min_reps}) — for transparency\n{'-'*70}")
    display_cols_e = ['Rank', 'EOS_id', 'Drug_name', 'Phenotype', 'Activity_score',
                      'Replicate_correlation', 'N_replicates',
                      'Target_name', 'MoA']
    display_cols_e = [c for c in display_cols_e if c in output_excluded.columns]
    display_e = output_excluded[display_cols_e].copy()
    for c in ['Target_name']:
        if c in display_e.columns:
            display_e[c] = display_e[c].fillna('').astype(str).apply(
                lambda s: s if len(s) < 50 else s[:47]+'...')
    for c in ['Activity_score', 'Replicate_correlation']:
        if c in display_e.columns:
            display_e[c] = display_e[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else 'NaN')
    print(display_e.to_string(index=False, max_colwidth=50))

# ============================================================
# Stats
# ============================================================
print(f"\n{'-'*70}\nStatistics (ROBUST hits only)\n{'-'*70}")
if len(output_robust) > 0:
    print(f"\nBy category:\n{output_robust['Hit_category'].value_counts().to_string()}")
    print(f"\nBy phenotype:\n{output_robust['Phenotype'].value_counts().to_string()}")
    if 'MoA' in output_robust.columns:
        print(f"\nBy MoA:")
        print(output_robust['MoA'].fillna('unknown').value_counts().to_string())
    if 'N_targets' in output_robust.columns:
        valid_n = output_robust['N_targets'].dropna()
        if len(valid_n) > 0:
            print(f"\nN_targets distribution among hits:")
            print(f"  Median: {valid_n.median():.0f}")
            print(f"  Range:  {valid_n.min():.0f} - {valid_n.max():.0f}")
            print(f"  Promiscuous (>20 targets): {(valid_n > 20).sum()} / {len(valid_n)}")

print(f"\n{'='*70}\n=== DONE ===\n{'='*70}")
print(f"\nROBUST hits (downstream): {out_path}")
if len(output_excluded) > 0:
    print(f"EXCLUDED hits (audit):    {HITS_DIR / f'{plate}_hits_excluded_low_reps.csv'}")
print(f"\nSummary: {len(output_robust)} robust + {len(output_excluded)} excluded = {len(output_robust) + len(output_excluded)} total candidates")