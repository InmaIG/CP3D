"""
19_moa_recall_repurposing_hub.py
==================================
MoA recall benchmark using the Broad Drug Repurposing Hub primary-mechanism
annotation, replacing the noisy ChEMBL Target_name list used in scripts 17-18.

The Broad Drug Repurposing Hub (Corsello et al., Nat Med 2017) provides
manually curated primary-mechanism (`moa`) and primary-target (`target`)
annotation for ~10,000 drugs and chemical probes, used as the standard
ground truth in published Cell Painting MoA recall benchmarks
(Pahl et al., Cell Chem Biol 2023; Trapotsi et al., Commun Biol 2025;
JUMP-CP Consortium 2024).

Pipeline
--------
1. Load Repurposing Hub `drugs` and `samples` tables.
2. Link drugs <-> samples by `pert_iname` to obtain (InChIKey, moa, target).
3. Match library compounds to Repurposing Hub by InChIKey (with skeleton
   fallback for stereochemistry differences) and by drug name (case-insensitive).
4. Filter to `moa` classes containing >= MIN_CLASS_SIZE matched compounds.
5. Compute pair-level AUC and mAP (CellProfiler vs DINOv2 feature spaces).
6. Per-class AUC breakdown.
7. Comparison against the raw ChEMBL annotation (script 17/18) if available.

Inputs
------
data/external/repurposing_drugs.txt
data/external/repurposing_samples.txt
data/annotated/cp3d_library_annotated.csv
data/embeddings/embeddings_per_compound.parquet
data/processed/consensus_C2386.csv, C2387.csv, C2388.csv

Output (results/moa_recall_repurposing_hub/)
--------------------------------------------
library_repurposing_mapping.csv
moa_recall_summary.csv
per_class_recall.csv
summary.txt
figures/
    roc_curves.png
    per_class_auc_bars.png
    comparison_with_chembl.png  (if scripts 17/18 outputs exist)

Usage
-----
    python 19_moa_recall_repurposing_hub.py
    python 19_moa_recall_repurposing_hub.py --min_class_size 3
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['pdf.fonttype'] = 42

BASE = Path(r'C:\Users\Ianezi\Documents\CP3D')
ANALYSIS = BASE / 'analysis'
EXT = ANALYSIS / 'data' / 'external'
LIB = ANALYSIS / 'data' / 'annotated' / 'cp3d_library_annotated.csv'
EMB = ANALYSIS / 'data' / 'embeddings' / 'embeddings_per_compound.parquet'
PROC = ANALYSIS / 'data' / 'processed'
OUT = ANALYSIS / 'results' / 'moa_recall_repurposing_hub'
FIG = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CONSENSUS_FILES = {
    'C2386': PROC / 'consensus_C2386.csv',
    'C2387': PROC / 'consensus_C2387.csv',
    'C2388': PROC / 'consensus_C2388.csv',
}

parser = argparse.ArgumentParser()
parser.add_argument('--drugs_file', default=None,
                    help='Path to repurposing_drugs.txt (default: auto-detect in data/external/)')
parser.add_argument('--samples_file', default=None,
                    help='Path to repurposing_samples.txt')
parser.add_argument('--min_class_size', type=int, default=3,
                    help='Minimum compounds per MoA class to evaluate (default 3).')
parser.add_argument('--k_list', nargs='+', type=int, default=[1, 5, 10, 25])
args = parser.parse_args()

print(f"\n{'='*70}\n=== MoA recall — Broad Drug Repurposing Hub annotation ===\n{'='*70}\n")

# ---------------------------------------------------------------------------
# 1. Load Repurposing Hub files
# ---------------------------------------------------------------------------
print(f"{'-'*70}\n1. Loading Drug Repurposing Hub files\n{'-'*70}")
drugs_path = Path(args.drugs_file) if args.drugs_file else None
samples_path = Path(args.samples_file) if args.samples_file else None
if drugs_path is None:
    cands = list(EXT.glob('repurposing_drugs*.txt')) + list(EXT.glob('repurposing_drugs*.tsv'))
    if not cands:
        print(f"  ERROR: no repurposing_drugs*.txt in {EXT}")
        print(f"  Download from https://clue.io/repurposing")
        sys.exit(1)
    drugs_path = cands[0]
if samples_path is None:
    cands = list(EXT.glob('repurposing_samples*.txt')) + list(EXT.glob('repurposing_samples*.tsv'))
    if not cands:
        print(f"  WARNING: no repurposing_samples*.txt found. InChIKey matching disabled.")
        samples_path = None
    else:
        samples_path = cands[0]
print(f"  Drugs file:    {drugs_path.name}")
print(f"  Samples file:  {samples_path.name if samples_path else '(none)'}")

# The files have header comments starting with '!'; skip them
def smart_read_repurposing(path):
    # Auto-detect header row (skip rows starting with '!')
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        skip = 0
        for line in f:
            if line.startswith('!') or line.strip() == '':
                skip += 1
            else:
                break
    return pd.read_csv(path, sep='\t', skiprows=skip, encoding='utf-8',
                       on_bad_lines='skip')

drugs = smart_read_repurposing(drugs_path)
print(f"  Drugs table: {drugs.shape}, columns: {list(drugs.columns)[:10]}")

if samples_path:
    samples = smart_read_repurposing(samples_path)
    print(f"  Samples table: {samples.shape}, columns: {list(samples.columns)[:10]}")
else:
    samples = None

# ---------------------------------------------------------------------------
# 2. Build pert_iname -> (moa, target) mapping
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n2. Building MoA / target lookup\n{'-'*70}")
# Find columns
def find_col(df, patterns):
    for c in df.columns:
        cl = str(c).lower().strip()
        for p in patterns:
            if p.lower() == cl:
                return c
    # Loose match
    for c in df.columns:
        cl = str(c).lower().strip()
        for p in patterns:
            if p.lower() in cl:
                return c
    return None

iname_col_drugs = find_col(drugs, ['pert_iname', 'name', 'drug_name'])
moa_col = find_col(drugs, ['moa', 'mechanism', 'mechanism_of_action'])
target_col = find_col(drugs, ['target', 'gene'])
phase_col = find_col(drugs, ['clinical_phase', 'phase'])
print(f"  Drugs columns: pert_iname={iname_col_drugs}, moa={moa_col}, "
      f"target={target_col}, phase={phase_col}")
if not iname_col_drugs or not moa_col:
    print(f"  ERROR: required columns not found. Available: {list(drugs.columns)}")
    sys.exit(1)

drugs = drugs.rename(columns={iname_col_drugs: 'pert_iname',
                                moa_col: 'moa'})
if target_col:
    drugs = drugs.rename(columns={target_col: 'target'})
if phase_col:
    drugs = drugs.rename(columns={phase_col: 'clinical_phase'})

drugs['_pname_norm'] = drugs['pert_iname'].fillna('').astype(str).str.upper().str.strip()
print(f"  Total drugs with MoA: {drugs['moa'].notna().sum()}")
print(f"\n  Top MoAs in Repurposing Hub (sample):")
print(drugs['moa'].value_counts().head(10).to_string())

# ---------------------------------------------------------------------------
# 3. Cross library with Repurposing Hub
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n3. Matching library compounds to Repurposing Hub\n{'-'*70}")
lib = pd.read_csv(LIB)
print(f"  Library size: {len(lib)}")
lib['_drug_norm'] = lib['Drug_name'].fillna('').astype(str).str.upper().str.strip()

# Pass 1: name-based match (drug name -> moa)
name_to_moa = (drugs.dropna(subset=['moa'])
                     .drop_duplicates(subset=['_pname_norm'])
                     .set_index('_pname_norm')[['moa']])
if 'target' in drugs.columns:
    name_to_moa['target'] = drugs.drop_duplicates(subset=['_pname_norm']).set_index('_pname_norm')['target']
merged = lib.merge(name_to_moa, left_on='_drug_norm', right_index=True, how='left')
n_by_name = merged['moa'].notna().sum()
print(f"  Matched by drug name:     {n_by_name}/{len(lib)} "
      f"({100*n_by_name/len(lib):.1f}%)")

# Pass 2: InChIKey-based match (via samples file)
n_added_ik = 0
if samples is not None:
    ik_col_samples = find_col(samples, ['InChIKey', 'inchi_key'])
    iname_col_samples = find_col(samples, ['pert_iname', 'name'])
    if ik_col_samples and iname_col_samples:
        samples = samples.rename(columns={ik_col_samples: 'InChIKey',
                                          iname_col_samples: 'pert_iname'})
        # Map InChIKey -> pert_iname -> moa
        sample_to_iname = (samples.dropna(subset=['InChIKey', 'pert_iname'])
                                   .drop_duplicates(subset=['InChIKey'])
                                   .assign(_pname_norm=lambda d:
                                            d['pert_iname'].fillna('').astype(str).str.upper().str.strip())
                                   .set_index('InChIKey')['_pname_norm'].to_dict())

        # Try each InChIKey variant in library
        ik_skel_to_iname = {}
        for ik, pname in sample_to_iname.items():
            ik_skel_to_iname.setdefault(str(ik).split('-')[0], pname)

        for ik_col in ['InChIKey', 'InChIKey_rdkit', 'InChIKey_moa']:
            if ik_col not in merged.columns: continue
            missing = merged['moa'].isna()
            for idx in merged[missing].index:
                ik = merged.at[idx, ik_col]
                if pd.notna(ik):
                    # Exact
                    if ik in sample_to_iname:
                        pname = sample_to_iname[ik]
                        if pname in name_to_moa.index:
                            merged.at[idx, 'moa'] = name_to_moa.loc[pname, 'moa']
                            if 'target' in name_to_moa.columns:
                                merged.at[idx, 'target'] = name_to_moa.loc[pname, 'target']
                            n_added_ik += 1
                            continue
                    # Skeleton fallback
                    skel = str(ik).split('-')[0]
                    if skel in ik_skel_to_iname:
                        pname = ik_skel_to_iname[skel]
                        if pname in name_to_moa.index:
                            merged.at[idx, 'moa'] = name_to_moa.loc[pname, 'moa']
                            if 'target' in name_to_moa.columns:
                                merged.at[idx, 'target'] = name_to_moa.loc[pname, 'target']
                            n_added_ik += 1
                            continue
print(f"  +Matched by InChIKey:     +{n_added_ik}")
n_total = merged['moa'].notna().sum()
print(f"  TOTAL matched with MoA:   {n_total}/{len(lib)} ({100*n_total/len(lib):.1f}%)")

# ---------------------------------------------------------------------------
# 4. Determine eligible MoA classes
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n4. MoA class membership\n{'-'*70}")

def parse_moas(s):
    if pd.isna(s):
        return set()
    parts = [p.strip() for p in str(s).replace(';', ',').split(',')]
    return set(p for p in parts if p)

merged['_moa_set'] = merged['moa'].apply(parse_moas)
# Count members per MoA
from collections import Counter
moa_counts = Counter()
for s in merged['_moa_set']:
    for m in s:
        moa_counts[m] += 1
eligible_moas = {m for m, n in moa_counts.items() if n >= args.min_class_size}
print(f"  Distinct MoAs in matched compounds:   {len(moa_counts)}")
print(f"  Eligible MoAs (>= {args.min_class_size} compounds): {len(eligible_moas)}")
print(f"\n  Top MoAs by membership:")
for m, n in sorted(moa_counts.items(), key=lambda x: -x[1])[:15]:
    eligible_mark = '*' if n >= args.min_class_size else ' '
    print(f"    {eligible_mark} {m}: {n}")

merged['_moa_eligible'] = merged['_moa_set'].apply(lambda s: s & eligible_moas)
analysis_set = merged[merged['_moa_eligible'].str.len() > 0].reset_index(drop=True)
print(f"\n  Compounds in at least one eligible MoA: {len(analysis_set)}")

# Save mapping
merged[['EOS_id', 'Drug_name', 'moa', 'target' if 'target' in merged.columns else 'moa']].to_csv(
    OUT / 'library_repurposing_mapping.csv', index=False)

if len(analysis_set) < 30:
    print(f"\n  WARNING: only {len(analysis_set)} compounds; MoA recall may be underpowered.")

# ---------------------------------------------------------------------------
# 5. Load feature spaces and compute distance matrices
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n5. Computing feature space distances\n{'-'*70}")
dfs = []
for plate, path in CONSENSUS_FILES.items():
    if path.exists():
        d = pd.read_csv(path)
        d = d[d['Metadata_Well_type'] == 'compound'].copy()
        d = d.rename(columns={'Metadata_Compound': 'EOS_id'})
        dfs.append(d)
cp = pd.concat(dfs, ignore_index=True)
feat_cp = [c for c in cp.columns if not c.startswith('Metadata_')
            and c != 'EOS_id' and pd.api.types.is_numeric_dtype(cp[c])]
cp_unique = cp.groupby('EOS_id')[feat_cp].mean().reset_index()
emb_dn = pd.read_parquet(EMB)
feat_dn = [c for c in emb_dn.columns if c != 'EOS_id']

analytic = analysis_set.merge(cp_unique, on='EOS_id', how='inner').merge(emb_dn, on='EOS_id', how='inner')
print(f"  Analytical set (in MoA + CP + DINOv2): {len(analytic)} compounds")

if len(analytic) < 20:
    print(f"  ERROR: too few compounds with all three sources. Cannot run benchmark.")
    sys.exit(1)

def zscore_distance(df, feat):
    X = df[feat].values.astype(float)
    X = np.nan_to_num(X, nan=0.0)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return squareform(pdist(Xz, metric='euclidean'))

D_cp = zscore_distance(analytic, feat_cp)
D_dn = zscore_distance(analytic, feat_dn)

n = len(analytic)
iu = np.triu_indices(n, k=1)
moa_sets = analytic['_moa_eligible'].tolist()
labels = np.array([1 if (moa_sets[i] & moa_sets[j]) else 0
                    for i, j in zip(iu[0], iu[1])])
n_pos = int(labels.sum())
print(f"  Pairs analyzed: {len(labels)}, same-MoA positives: {n_pos} "
      f"({100*n_pos/len(labels):.1f}%)")

# ---------------------------------------------------------------------------
# 6. Pair-level AUC + mAP
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n6. Pair-level MoA recall metrics\n{'-'*70}")
results = []
roc_data = {}
for name, D in [('CellProfiler', D_cp), ('DINOv2', D_dn)]:
    pair_dist = D[iu]
    scores = -pair_dist
    auc = roc_auc_score(labels, scores)
    map_ = average_precision_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_data[name] = (fpr, tpr, auc)
    print(f"  {name:13s}: AUC = {auc:.3f},  mAP = {map_:.3f}")
    results.append({'feature_space': name, 'metric': 'pair_AUC', 'value': auc})
    results.append({'feature_space': name, 'metric': 'pair_mAP', 'value': map_})

# Top-K recall
def topk(D, sets, K_list):
    rows = []
    for i in range(len(sets)):
        own = sets[i]
        order = np.argsort(D[i]); order = order[order != i]
        same = np.array([1 if (sets[j] & own) else 0 for j in order])
        if same.sum() == 0:
            continue
        row = {'idx': i}
        for K in K_list:
            row[f'recall@{K}'] = same[:K].sum() / same.sum()
            row[f'precision@{K}'] = same[:K].sum() / K
        rows.append(row)
    return pd.DataFrame(rows)

for name, D in [('CellProfiler', D_cp), ('DINOv2', D_dn)]:
    pc = topk(D, moa_sets, args.k_list)
    print(f"\n  {name}:")
    for K in args.k_list:
        rec = pc[f'recall@{K}'].mean()
        prec = pc[f'precision@{K}'].mean()
        print(f"    Recall@{K:>2} = {rec:.3f},  Precision@{K:>2} = {prec:.3f}")
        results.append({'feature_space': name, 'metric': f'recall@{K}', 'value': rec})
        results.append({'feature_space': name, 'metric': f'precision@{K}', 'value': prec})

pd.DataFrame(results).to_csv(OUT / 'moa_recall_summary.csv', index=False)

# ---------------------------------------------------------------------------
# 7. Per-class AUC
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n7. Per-class AUC\n{'-'*70}")
class_rows = []
for m in sorted(eligible_moas):
    cl = np.array([1 if (m in moa_sets[i] and m in moa_sets[j]) else 0
                    for i, j in zip(iu[0], iu[1])])
    n_match = int(cl.sum())
    if n_match < 1: continue
    n_members = sum(1 for s in moa_sets if m in s)
    auc_cp = roc_auc_score(cl, -D_cp[iu])
    auc_dn = roc_auc_score(cl, -D_dn[iu])
    class_rows.append({'moa': m, 'n_members': n_members, 'n_same_pairs': n_match,
                        'AUC_CellProfiler': auc_cp, 'AUC_DINOv2': auc_dn})
class_df = pd.DataFrame(class_rows).sort_values('AUC_CellProfiler', ascending=False)
class_df.to_csv(OUT / 'per_class_recall.csv', index=False)
print(class_df.head(20).to_string(index=False))

# ---------------------------------------------------------------------------
# 8. Plots
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n8. Plots\n{'-'*70}")

# ROC
fig, ax = plt.subplots(figsize=(7, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    color = '#1F77B4' if name == 'CellProfiler' else '#D7263D'
    ax.plot(fpr, tpr, color=color, lw=2.2, label=f'{name} (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], '--', color='#888', lw=0.7, alpha=0.7, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'MoA recall — Broad Drug Repurposing Hub annotation\n'
              f'({n_pos} same-MoA pairs, {len(eligible_moas)} classes, '
              f'{len(analytic)} compounds)')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(FIG / 'roc_curves.png', dpi=200, bbox_inches='tight')
plt.close()

# Per-class
if len(class_df):
    show = class_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(show))))
    y = np.arange(len(show))
    ax.barh(y - 0.18, show['AUC_CellProfiler'], height=0.36, color='#1F77B4',
             edgecolor='black', label='CellProfiler')
    ax.barh(y + 0.18, show['AUC_DINOv2'], height=0.36, color='#D7263D',
             edgecolor='black', label='DINOv2')
    ax.axvline(0.5, ls='--', color='#888', lw=0.7, alpha=0.7, label='Random')
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['moa'][:45]} (n={int(r['n_members'])})"
                         for _, r in show.iterrows()], fontsize=8.5)
    ax.set_xlabel('Pair-level AUC')
    ax.set_title('MoA recall by Broad Drug Repurposing Hub MoA class (top 20)')
    ax.legend(loc='lower right')
    ax.set_xlim(0.3, 1.0)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG / 'per_class_auc_bars.png', dpi=200, bbox_inches='tight')
    plt.close()

# Comparison with ChEMBL crudo (if exists)
chembl_csv = ANALYSIS / 'results' / 'moa_recall' / 'moa_recall_summary.csv'
if chembl_csv.exists():
    raw = pd.read_csv(chembl_csv)
    raw_cp = raw[(raw.feature_space=='CellProfiler') & (raw.metric=='pair_AUC')]['value'].iloc[0]
    raw_dn = raw[(raw.feature_space=='DINOv2') & (raw.metric=='pair_AUC')]['value'].iloc[0]
    rh_cp = next(r['value'] for r in results if r['feature_space']=='CellProfiler' and r['metric']=='pair_AUC')
    rh_dn = next(r['value'] for r in results if r['feature_space']=='DINOv2' and r['metric']=='pair_AUC')

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2); width = 0.35
    ax.bar(x - width/2, [raw_cp, raw_dn], width, color='#888', edgecolor='black',
            label='ChEMBL Target_name (raw)')
    ax.bar(x + width/2, [rh_cp, rh_dn], width, color='#4DAF4A', edgecolor='black',
            label='Repurposing Hub MoA (curated)')
    ax.axhline(0.5, ls='--', color='#888', lw=0.7, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(['CellProfiler', 'DINOv2'])
    ax.set_ylabel('Pair-level AUC')
    ax.set_title('Effect of annotation source on MoA recall')
    ax.legend()
    for i, (rv, cv) in enumerate(zip([raw_cp, raw_dn], [rh_cp, rh_dn])):
        ax.text(i - width/2, rv + 0.01, f'{rv:.3f}', ha='center', fontsize=10)
        ax.text(i + width/2, cv + 0.01, f'{cv:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG / 'comparison_with_chembl.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ChEMBL vs Repurposing Hub comparison saved.")

print(f"  Figures saved in: {FIG}")

# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------
summary = OUT / 'summary.txt'
auc_cp_v = next(r['value'] for r in results if r['feature_space']=='CellProfiler' and r['metric']=='pair_AUC')
auc_dn_v = next(r['value'] for r in results if r['feature_space']=='DINOv2' and r['metric']=='pair_AUC')
with open(summary, 'w', encoding='utf-8') as f:
    f.write("MoA recall — Broad Drug Repurposing Hub annotation\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Source: Corsello et al., Nat Med 2017 (Broad CMap / Drug Repurposing Hub)\n")
    f.write(f"Library compounds mapped to Repurposing Hub MoA: {n_total}/{len(lib)} "
            f"({100*n_total/len(lib):.1f}%)\n")
    f.write(f"Compounds in eligible MoA classes: {len(analysis_set)}\n")
    f.write(f"Compounds in CP + DINOv2 + MoA (analytical):  {len(analytic)}\n")
    f.write(f"Same-MoA positive pairs: {n_pos}\n\n")
    f.write("PAIR-LEVEL METRICS\n")
    f.write("-" * 60 + "\n")
    f.write(f"  CellProfiler:  AUC = {auc_cp_v:.3f}\n")
    f.write(f"  DINOv2:        AUC = {auc_dn_v:.3f}\n\n")
    f.write("TOP MoA classes\n")
    f.write("-" * 60 + "\n")
    f.write(class_df.head(20).to_string(index=False))

print(f"\n  Summary: {summary}")
print(f"\n{'='*70}\nDone. Outputs in: {OUT}\n{'='*70}\n")
