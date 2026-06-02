"""
18_moa_recall_curated.py
==========================
Mechanism-of-action recall benchmark using curated target families.

This script complements script 17 by replacing the raw ChEMBL Target_name
annotation (which contains off-target and pharmacokinetic associations such as
CYP450 metabolism — diluting MoA recall to AUC ≈ 0.5) with a manually curated
set of ~15 biologically meaningful target families. Each compound's
Target_name field is matched against the curated patterns to assign it to
one or more functional families, yielding cleaner ground truth.

Curated families used here are derived from:
 - Cell Painting MoA cluster literature (Pahl et al., Cell Chem Biol 2023)
 - Pathways relevant to HepG2 / hepatocyte biology (Hippo/STK, HSP90 client
   folding, CYP metabolism)
 - Target categories enriched in the 25 robust 3D hits

Inputs (same as script 17)
--------------------------
data/annotated/cp3d_library_annotated.csv
data/embeddings/embeddings_per_compound.parquet
data/processed/consensus_C2386.csv, consensus_C2387.csv, consensus_C2388.csv

Outputs (results/moa_recall_curated/)
-------------------------------------
moa_recall_curated_summary.csv
per_curated_class_recall.csv
group_membership.csv
summary.txt
figures/
    roc_curves_curated.png
    per_class_auc_bars.png
    raw_vs_curated_comparison.png

Usage
-----
    python 18_moa_recall_curated.py
    python 18_moa_recall_curated.py --k_list 1 5 10
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(r'C:\Users\Ianezi\Documents\CP3D')
ANALYSIS = BASE / 'analysis'
LIB = ANALYSIS / 'data' / 'annotated' / 'cp3d_library_annotated.csv'
EMB = ANALYSIS / 'data' / 'embeddings' / 'embeddings_per_compound.parquet'
PROC = ANALYSIS / 'data' / 'processed'
OUT = ANALYSIS / 'results' / 'moa_recall_curated'
FIG = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CONSENSUS_FILES = {
    'C2386': PROC / 'consensus_C2386.csv',
    'C2387': PROC / 'consensus_C2387.csv',
    'C2388': PROC / 'consensus_C2388.csv',
}

# ---------------------------------------------------------------------------
# Curated target families (patterns are matched as case-insensitive substrings
# against Target_name strings, which may be comma- or semicolon-separated lists)
# ---------------------------------------------------------------------------
CURATED_GROUPS = {
    # --- Cytoskeleton -------------------------------------------------------
    'Tubulin / Microtubule':
        ['tubulin'],

    # --- DNA-related ---------------------------------------------------------
    'DNA Topoisomerase':
        ['topoisomerase'],

    'DNA Polymerase / Repair':
        ['dna polymerase', 'apurinic', 'apyrimidinic site',
         'xpd helicase', 'helicase subunit xpd', 'endonuclease 4'],

    # --- Chaperones / proteostasis ------------------------------------------
    'HSP90 Chaperone System':
        ['heat shock protein hsp90', 'hsp 90-alpha', 'hsp 90-beta',
         'endoplasmin', 'hsp90'],

    'Proteasome':
        ['proteasome'],

    # --- Kinase signaling ---------------------------------------------------
    'CDK Family (cell cycle)':
        ['cyclin-dependent kinase'],

    'STE20 / STK family (Hippo pathway)':
        ['serine/threonine-protein kinase 24',
         'serine/threonine-protein kinase 4',
         'serine/threonine-protein kinase 25',
         'serine/threonine-protein kinase 3',
         'serine/threonine-protein kinase stk11',
         'serine/threonine-protein kinase n1',
         'mitogen-activated protein kinase kinase kinase kinase'],

    'p38 MAPK pathway':
        ['mitogen-activated protein kinase 11', 'mitogen-activated protein kinase 12',
         'mitogen-activated protein kinase 13', 'mitogen-activated protein kinase 14',
         'p38'],

    'EGFR / ErbB family':
        ['epidermal growth factor receptor', 'receptor tyrosine-protein kinase erbb',
         'her2', 'her3', 'her4'],

    'Aurora / PLK kinases':
        ['aurora', 'polo-like kinase', 'serine/threonine-protein kinase plk'],

    'JAK / STAT pathway':
        ['janus kinase', 'jak1', 'jak2', 'jak3', 'tyrosine-protein kinase jak'],

    'PI3K / AKT / mTOR':
        ['phosphatidylinositol 4,5-bisphosphate 3-kinase',
         'serine/threonine-protein kinase mtor',
         'rac-alpha serine/threonine-protein kinase',
         'akt'],

    # --- Epigenetics --------------------------------------------------------
    'Histone / Epigenetics':
        ['histone-lysine n-methyltransferase', 'menin', 'kmt2',
         'histone deacetylase', 'dna (cytosine-5)-methyltransferase'],

    # --- Apoptosis / cell death --------------------------------------------
    'Caspases / Apoptosis':
        ['caspase'],

    # --- Nuclear receptors --------------------------------------------------
    'Steroid Nuclear Receptors':
        ['estrogen receptor', 'androgen receptor', 'glucocorticoid receptor',
         'progesterone receptor'],
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--k_list', nargs='+', type=int, default=[1, 5, 10, 25])
parser.add_argument('--target_col', default='Target_name')
parser.add_argument('--min_class_size', type=int, default=3,
                    help='Minimum compounds per curated group to evaluate (default 3).')
args = parser.parse_args()

print(f"\n{'='*70}\n=== MoA recall — CURATED target families ===\n{'='*70}\n")

# ---------------------------------------------------------------------------
# 1. Load + merge
# ---------------------------------------------------------------------------
print(f"{'-'*70}\n1. Loading datasets\n{'-'*70}")
lib = pd.read_csv(LIB)

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
emb = pd.read_parquet(EMB)
feat_dn = [c for c in emb.columns if c != 'EOS_id']

common = lib.merge(cp_unique, on='EOS_id', how='inner').merge(emb, on='EOS_id', how='inner')
print(f"  Library:             {len(lib)} compounds")
print(f"  CellProfiler matrix: {cp_unique.shape}")
print(f"  DINOv2 matrix:       {emb.shape}")
print(f"  Merged inner:        {len(common)} compounds")

# ---------------------------------------------------------------------------
# 2. Assign curated groups to each compound
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n2. Assigning curated target groups\n{'-'*70}")

def assign_curated(target_string):
    if pd.isna(target_string):
        return set()
    s = str(target_string).lower()
    groups = set()
    for group_name, patterns in CURATED_GROUPS.items():
        for pat in patterns:
            if pat.lower() in s:
                groups.add(group_name)
                break
    return groups

common['_curated_groups'] = common[args.target_col].apply(assign_curated)

# Membership statistics
group_counts = {g: 0 for g in CURATED_GROUPS}
for gset in common['_curated_groups']:
    for g in gset:
        group_counts[g] += 1
group_counts_df = pd.DataFrame([
    {'group': g, 'n_compounds': n} for g, n in group_counts.items()
]).sort_values('n_compounds', ascending=False)
group_counts_df.to_csv(OUT / 'group_membership.csv', index=False)

print(f"\n  Curated group membership counts:")
for _, r in group_counts_df.iterrows():
    eligible = "✓" if r['n_compounds'] >= args.min_class_size else "✗ (skipped, too small)"
    print(f"    {r['group']:38s} n = {r['n_compounds']:>3}   {eligible}")

eligible_groups = {g for g, n in group_counts.items() if n >= args.min_class_size}
print(f"\n  Eligible groups (>= {args.min_class_size} compounds): {len(eligible_groups)}")

# Filter compound groups to eligible only
common['_curated_eligible'] = common['_curated_groups'].apply(
    lambda s: s & eligible_groups)
n_with_curated = (common['_curated_eligible'].str.len() > 0).sum()
print(f"  Compounds in at least one eligible group: {n_with_curated}/{len(common)}")

# Analytical set
analysis_set = common[common['_curated_eligible'].str.len() > 0].reset_index(drop=True)
print(f"  Analytical subset: {len(analysis_set)} compounds")

# ---------------------------------------------------------------------------
# 3. Distance matrices and pair labels
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n3. Computing distance matrices and pair labels\n{'-'*70}")

def zscore_distance(df, feat):
    X = df[feat].values.astype(float)
    X = np.nan_to_num(X, nan=0.0)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return squareform(pdist(Xz, metric='euclidean'))

D_cp = zscore_distance(analysis_set, feat_cp)
D_dn = zscore_distance(analysis_set, feat_dn)

n = len(analysis_set)
iu = np.triu_indices(n, k=1)
group_sets = analysis_set['_curated_eligible'].tolist()
labels = np.array([1 if (group_sets[i] & group_sets[j]) else 0
                    for i, j in zip(iu[0], iu[1])])
n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
print(f"  Pairs analyzed:              {len(labels)}")
print(f"    Same-curated-group (pos):   {n_pos} ({100*n_pos/len(labels):.2f}%)")
print(f"    Different (neg):            {n_neg}")

# ---------------------------------------------------------------------------
# 4. Pair-level AUC + mAP
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n4. Pair-level MoA recall\n{'-'*70}")
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

# ---------------------------------------------------------------------------
# 5. Top-K recall
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n5. Top-K recall per compound\n{'-'*70}")
def topk_metrics(D, group_sets, K_list):
    per = []
    for i in range(len(group_sets)):
        own = group_sets[i]
        order = np.argsort(D[i]); order = order[order != i]
        same = np.array([1 if (group_sets[j] & own) else 0 for j in order])
        if same.sum() == 0:
            continue
        row = {'compound_idx': i, 'n_same_total': int(same.sum())}
        for K in K_list:
            row[f'recall@{K}'] = same[:K].sum() / same.sum()
            row[f'precision@{K}'] = same[:K].sum() / K
        per.append(row)
    return pd.DataFrame(per)

for name, D in [('CellProfiler', D_cp), ('DINOv2', D_dn)]:
    pc = topk_metrics(D, group_sets, args.k_list)
    print(f"\n  {name}:")
    for K in args.k_list:
        rec = pc[f'recall@{K}'].mean()
        prec = pc[f'precision@{K}'].mean()
        print(f"    Recall@{K:>2} = {rec:.3f},  Precision@{K:>2} = {prec:.3f}")
        results.append({'feature_space': name, 'metric': f'recall@{K}', 'value': rec})
        results.append({'feature_space': name, 'metric': f'precision@{K}', 'value': prec})

results_df = pd.DataFrame(results)
results_df.to_csv(OUT / 'moa_recall_curated_summary.csv', index=False)

# ---------------------------------------------------------------------------
# 6. Per-curated-class AUC
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n6. Per-class AUC\n{'-'*70}")
class_rows = []
for g in sorted(eligible_groups):
    class_labels = np.array([
        1 if (g in group_sets[i] and g in group_sets[j]) else 0
        for i, j in zip(iu[0], iu[1])])
    n_match = int(class_labels.sum())
    if n_match < 1:
        continue
    n_members = sum(1 for gs in group_sets if g in gs)
    auc_cp = roc_auc_score(class_labels, -D_cp[iu])
    auc_dn = roc_auc_score(class_labels, -D_dn[iu])
    class_rows.append({
        'curated_group': g, 'n_members': n_members, 'n_same_pairs': n_match,
        'AUC_CellProfiler': auc_cp, 'AUC_DINOv2': auc_dn,
    })
class_df = pd.DataFrame(class_rows).sort_values('AUC_CellProfiler', ascending=False)
class_df.to_csv(OUT / 'per_curated_class_recall.csv', index=False)
print(class_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n7. Plots\n{'-'*70}")

# Plot 1: ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
colors_map = {'CellProfiler': '#1F77B4', 'DINOv2': '#D7263D'}
for name, (fpr, tpr, auc) in roc_data.items():
    ax.plot(fpr, tpr, color=colors_map[name], lw=2.2,
             label=f'{name} (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], '--', color='#888', lw=0.7, alpha=0.7,
         label='Random (AUC = 0.5)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title(f'MoA recall — curated target families\n'
              f'({n_pos} positive pairs, {len(eligible_groups)} target groups, '
              f'{len(analysis_set)} compounds)')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(FIG / 'roc_curves_curated.png', dpi=200, bbox_inches='tight')
plt.close()

# Plot 2: Per-class AUC bars
if len(class_df):
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(class_df))))
    cd = class_df.iloc[::-1].copy()
    y = np.arange(len(cd))
    ax.barh(y - 0.18, cd['AUC_CellProfiler'], height=0.36, color='#1F77B4',
             edgecolor='black', label='CellProfiler')
    ax.barh(y + 0.18, cd['AUC_DINOv2'], height=0.36, color='#D7263D',
             edgecolor='black', label='DINOv2')
    ax.axvline(0.5, ls='--', color='#888', lw=0.7, alpha=0.7, label='Random')
    ax.set_yticks(y)
    labs = [f"{r['curated_group']} (n={int(r['n_members'])})"
             for _, r in cd.iterrows()]
    ax.set_yticklabels(labs, fontsize=9)
    ax.set_xlabel('Pair-level AUC')
    ax.set_title('MoA recall per curated target family')
    ax.legend(loc='lower right')
    ax.set_xlim(0.4, 1.0)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG / 'per_class_auc_bars.png', dpi=200, bbox_inches='tight')
    plt.close()

# Plot 3: Raw vs curated AUC comparison (if script 17 results exist)
raw_csv = ANALYSIS / 'results' / 'moa_recall' / 'moa_recall_summary.csv'
if raw_csv.exists():
    raw_df = pd.read_csv(raw_csv)
    raw_auc_cp = raw_df[(raw_df.feature_space=='CellProfiler') & (raw_df.metric=='pair_AUC')]['value'].iloc[0]
    raw_auc_dn = raw_df[(raw_df.feature_space=='DINOv2') & (raw_df.metric=='pair_AUC')]['value'].iloc[0]
    cur_auc_cp = results_df[(results_df.feature_space=='CellProfiler') & (results_df.metric=='pair_AUC')]['value'].iloc[0]
    cur_auc_dn = results_df[(results_df.feature_space=='DINOv2') & (results_df.metric=='pair_AUC')]['value'].iloc[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, [raw_auc_cp, raw_auc_dn], width, color='#888',
            edgecolor='black', label=f'Raw Target_name annotation\n(n = 994 classes)')
    ax.bar(x + width/2, [cur_auc_cp, cur_auc_dn], width, color='#4DAF4A',
            edgecolor='black', label=f'Curated families\n(n = {len(eligible_groups)} classes)')
    ax.axhline(0.5, ls='--', color='#888', lw=0.7, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(['CellProfiler', 'DINOv2'])
    ax.set_ylabel('Pair-level AUC')
    ax.set_title('Annotation curation effect on MoA recall')
    ax.legend()
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for i, (raw, cur) in enumerate(zip([raw_auc_cp, raw_auc_dn], [cur_auc_cp, cur_auc_dn])):
        ax.text(i - width/2, raw + 0.005, f'{raw:.3f}', ha='center', fontsize=10)
        ax.text(i + width/2, cur + 0.005, f'{cur:.3f}', ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG / 'raw_vs_curated_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
print(f"  Figures saved in: {FIG}")

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
summary = OUT / 'summary.txt'
auc_cp_v = results_df[(results_df.feature_space=='CellProfiler') & (results_df.metric=='pair_AUC')]['value'].iloc[0]
auc_dn_v = results_df[(results_df.feature_space=='DINOv2') & (results_df.metric=='pair_AUC')]['value'].iloc[0]
with open(summary, 'w', encoding='utf-8') as f:
    f.write("MoA recall benchmark with curated target families\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Compounds in eligible curated groups: {len(analysis_set)}\n")
    f.write(f"Eligible curated families (>= {args.min_class_size} compounds): {len(eligible_groups)}\n")
    f.write(f"Same-family pairs:                    {n_pos} / {len(labels)} "
            f"({100*n_pos/len(labels):.1f}%)\n\n")
    f.write("Pair-level AUC and mAP:\n")
    f.write(f"  CellProfiler:  AUC = {auc_cp_v:.3f}\n")
    f.write(f"  DINOv2:        AUC = {auc_dn_v:.3f}\n\n")
    f.write("Curated group membership (counts):\n")
    f.write(group_counts_df.to_string(index=False))
    f.write("\n\nPer-class AUC:\n")
    f.write(class_df.to_string(index=False))
print(f"\n  Summary: {summary}")
print(f"\n{'='*70}\nDone. Outputs in: {OUT}\n{'='*70}\n")
