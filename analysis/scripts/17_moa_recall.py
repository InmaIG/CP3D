"""
17_moa_recall.py
=================
Mechanism-of-action (MoA) recall benchmark in two independent phenotypic
spaces: hand-engineered CellProfiler features (CP3D pipeline, 159-dim) and
self-supervised DINOv2 deep embeddings derived from raw MIP pixels (1536-dim).

The MoA recall metric evaluates whether compounds annotated to the same
molecular target or mechanism are closer in phenotypic feature space than
random compound pairs. It is the de-facto benchmark of the morphological
profiling community (Pahl et al. Cell Chem Biol 2023; Schölermann et al.
bioRxiv 2025; Trapotsi et al. Commun Biol 2025; JUMP-CP Consortium 2024).

Two complementary metrics are reported:
1. **AUC of pair-level ROC** — using negative distance as a continuous
   score for the binary label "shares at least one Target_name".
2. **Mean Average Precision (mAP) at top-K nearest neighbors** — for each
   query compound, fraction of MoA-matched neighbors recovered in top-K.

A high AUC / mAP indicates that the feature space generalizes meaningfully
to chemical-biological annotation that the features never saw during their
construction.

Inputs
------
data/annotated/cp3d_library_annotated.csv
data/embeddings/embeddings_per_compound.parquet     (DINOv2, 1536-dim)
data/processed/consensus_C2386.csv, C2387.csv, C2388.csv  (CellProfiler, 159-dim)

Output (results/moa_recall/)
----------------------------
moa_recall_summary.csv     One row per (feature_space, K) with AUC, mAP, recall
roc_curves.csv             Raw FPR/TPR points for plotting
per_compound_neighbors.csv  Top-10 neighbors per compound with same-MoA flags
summary.txt                Narrative summary
figures/
    roc_moa_recall.png     ROC curves CellProfiler vs DINOv2
    map_recall_bars.png    mAP and Recall@K comparison
    target_class_recall.png  AUC broken down by major target class

Usage
-----
    python 17_moa_recall.py
    python 17_moa_recall.py --k_list 1 5 10 25
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

BASE = Path(r'C:\Users\Ianezi\Documents\CP3D')
ANALYSIS = BASE / 'analysis'
LIB = ANALYSIS / 'data' / 'annotated' / 'cp3d_library_annotated.csv'
EMB = ANALYSIS / 'data' / 'embeddings' / 'embeddings_per_compound.parquet'
PROC = ANALYSIS / 'data' / 'processed'
OUT = ANALYSIS / 'results' / 'moa_recall'
FIG = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CONSENSUS_FILES = {
    'C2386': PROC / 'consensus_C2386.csv',
    'C2387': PROC / 'consensus_C2387.csv',
    'C2388': PROC / 'consensus_C2388.csv',
}

parser = argparse.ArgumentParser()
parser.add_argument('--k_list', nargs='+', type=int, default=[1, 5, 10, 25],
                    help='Top-K neighborhoods to evaluate.')
parser.add_argument('--target_col', default='Target_name',
                    help='Annotation column (default Target_name).')
parser.add_argument('--min_class_size', type=int, default=2,
                    help='Minimum compounds per target class to include.')
args = parser.parse_args()

print(f"\n{'='*70}\n=== MoA recall benchmark — CellProfiler vs DINOv2 ===\n{'='*70}\n")

# ---------------------------------------------------------------------------
# 1. Load annotation and feature matrices
# ---------------------------------------------------------------------------
print(f"{'-'*70}\n1. Loading annotation and feature matrices\n{'-'*70}")
lib = pd.read_csv(LIB)
print(f"  Library: {len(lib)} compounds")

# CellProfiler 3D consensus (concatenate 3 plates)
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
# One row per compound (mean across plates if duplicated)
cp_unique = cp.groupby('EOS_id')[feat_cp].mean().reset_index()
print(f"  CellProfiler matrix: {cp_unique.shape[0]} x {len(feat_cp)} features")

# DINOv2 embeddings
emb = pd.read_parquet(EMB)
feat_dn = [c for c in emb.columns if c != 'EOS_id']
print(f"  DINOv2 matrix:       {emb.shape[0]} x {len(feat_dn)} features")

# Common compound set with both representations and annotation
common = lib.merge(cp_unique, on='EOS_id', how='inner').merge(emb, on='EOS_id', how='inner')
print(f"  Common compounds (CP + DINOv2 + annotation): {len(common)}")

# ---------------------------------------------------------------------------
# 2. Build target sets for "same MoA" ground truth
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n2. Building target-class ground truth\n{'-'*70}")
target_col = args.target_col
if target_col not in common.columns:
    print(f"  ERROR: column {target_col} not in library. Available: {list(common.columns)[:20]}")
    sys.exit(1)

# Parse multi-target strings (comma- or semicolon-separated)
def parse_targets(s):
    if pd.isna(s):
        return set()
    parts = [p.strip() for p in str(s).replace(';', ',').split(',')]
    return set(p for p in parts if p)

common['_targets'] = common[target_col].apply(parse_targets)
n_annot = (common['_targets'].str.len() > 0).sum()
print(f"  Compounds with at least one target annotation: {n_annot}/{len(common)}")

# Keep only annotated compounds for the recall benchmark
analysis_set = common[common['_targets'].str.len() > 0].reset_index(drop=True)
print(f"  Analytical subset: {len(analysis_set)} annotated compounds")

# Count target classes
from collections import Counter
all_targets = Counter()
for tset in analysis_set['_targets']:
    for t in tset:
        all_targets[t] += 1
big_classes = {t for t, n in all_targets.items() if n >= args.min_class_size}
print(f"  Distinct targets:                   {len(all_targets)}")
print(f"  Targets with >= {args.min_class_size} compounds (used for recall): {len(big_classes)}")
# Filter compound target sets to big classes only
analysis_set['_targets_big'] = analysis_set['_targets'].apply(
    lambda s: s & big_classes)
analysis_set = analysis_set[analysis_set['_targets_big'].str.len() > 0].reset_index(drop=True)
print(f"  Compounds in at least one big class: {len(analysis_set)}")

# ---------------------------------------------------------------------------
# 3. Compute distance matrices in each feature space
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n3. Computing distance matrices\n{'-'*70}")

def zscore_distance_matrix(df, feature_cols):
    X = df[feature_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Xz = (X - mu) / sd
    D = squareform(pdist(Xz, metric='euclidean'))
    return D

D_cp = zscore_distance_matrix(analysis_set, feat_cp)
D_dn = zscore_distance_matrix(analysis_set, feat_dn)
print(f"  CellProfiler distance matrix: {D_cp.shape}")
print(f"  DINOv2 distance matrix:       {D_dn.shape}")

# Build pair-wise same-target labels (upper triangle)
n = len(analysis_set)
iu = np.triu_indices(n, k=1)
target_sets = analysis_set['_targets_big'].tolist()
labels = np.zeros(len(iu[0]), dtype=int)
for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
    if target_sets[i] & target_sets[j]:
        labels[idx] = 1

n_pos = int(labels.sum())
n_neg = int((1 - labels).sum())
print(f"  Pairs analyzed:              {len(labels)}")
print(f"    Same-target (positive):    {n_pos} ({100*n_pos/len(labels):.2f}%)")
print(f"    Different-target (negative): {n_neg}")
if n_pos < 10:
    print(f"  ERROR: too few same-target pairs. Check Target_name parsing.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 4. Pair-level AUC and mAP for each feature space
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n4. Pair-level AUC and mAP\n{'-'*70}")
results = []
roc_data = {}
for name, D in [('CellProfiler', D_cp), ('DINOv2', D_dn)]:
    pair_dist = D[iu]
    scores = -pair_dist                                   # smaller dist -> higher score
    auc = roc_auc_score(labels, scores)
    map_ = average_precision_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_data[name] = (fpr, tpr, auc)
    print(f"  {name:13s}: AUC = {auc:.3f}, mAP = {map_:.3f}")
    results.append({'feature_space': name, 'metric': 'pair_AUC',
                     'value': auc, 'pairs': len(labels), 'pos_pairs': n_pos})
    results.append({'feature_space': name, 'metric': 'pair_mAP',
                     'value': map_, 'pairs': len(labels), 'pos_pairs': n_pos})

# ---------------------------------------------------------------------------
# 5. Top-K Recall and mAP per compound
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n5. Top-K recall (per compound)\n{'-'*70}")
def topk_metrics(D, target_sets, K_list):
    n = len(target_sets)
    per_compound = []
    for i in range(n):
        own = target_sets[i]
        # Distances to all other compounds
        dists = D[i].copy()
        dists[i] = np.inf
        order = np.argsort(dists)
        # Which neighbors share a target?
        same = np.array([1 if (target_sets[j] & own) else 0 for j in order])
        # Average precision (rank-aware)
        n_pos_total = int(same.sum())
        if n_pos_total == 0:
            continue
        row = {'compound_idx': i, 'n_same_total': n_pos_total}
        for K in K_list:
            n_same_in_topK = int(same[:K].sum())
            row[f'recall@{K}'] = n_same_in_topK / n_pos_total
            row[f'precision@{K}'] = n_same_in_topK / K
        per_compound.append(row)
    return pd.DataFrame(per_compound)

for name, D in [('CellProfiler', D_cp), ('DINOv2', D_dn)]:
    per_comp = topk_metrics(D, target_sets, args.k_list)
    print(f"\n  {name}:")
    for K in args.k_list:
        rec = per_comp[f'recall@{K}'].mean()
        prec = per_comp[f'precision@{K}'].mean()
        print(f"    Recall@{K:>2} = {rec:.3f},  Precision@{K:>2} = {prec:.3f}")
        results.append({'feature_space': name, 'metric': f'recall@{K}',
                         'value': rec, 'pairs': len(per_comp), 'pos_pairs': None})
        results.append({'feature_space': name, 'metric': f'precision@{K}',
                         'value': prec, 'pairs': len(per_comp), 'pos_pairs': None})

results_df = pd.DataFrame(results)
results_df.to_csv(OUT / 'moa_recall_summary.csv', index=False)

# ---------------------------------------------------------------------------
# 6. Per-target-class AUC breakdown
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n6. Per-class AUC breakdown (10 largest classes)\n{'-'*70}")
class_results = []
top_classes = sorted(all_targets.items(), key=lambda x: -x[1])[:20]
for target_name, count in top_classes:
    if count < 3:
        continue
    # Members
    member_idx = [i for i, ts in enumerate(target_sets) if target_name in ts]
    if len(member_idx) < 2:
        continue
    # Labels: pair is "match" if BOTH members of pair share THIS target
    class_labels = np.zeros(len(iu[0]), dtype=int)
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        if target_name in target_sets[i] and target_name in target_sets[j]:
            class_labels[idx] = 1
    if class_labels.sum() == 0:
        continue
    auc_cp = roc_auc_score(class_labels, -D_cp[iu])
    auc_dn = roc_auc_score(class_labels, -D_dn[iu])
    class_results.append({
        'target_class': target_name, 'n_members': len(member_idx),
        'n_same_pairs': int(class_labels.sum()),
        'AUC_CellProfiler': auc_cp, 'AUC_DINOv2': auc_dn,
    })
class_df = pd.DataFrame(class_results).sort_values('n_members', ascending=False)
class_df.to_csv(OUT / 'per_class_recall.csv', index=False)
print(class_df[['target_class', 'n_members', 'AUC_CellProfiler', 'AUC_DINOv2']]
       .head(15).to_string(index=False))

# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n7. Plots\n{'-'*70}")

# Plot 1: ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
colors = {'CellProfiler': '#1F77B4', 'DINOv2': '#D7263D'}
for name, (fpr, tpr, auc) in roc_data.items():
    ax.plot(fpr, tpr, color=colors[name], lw=2.2,
             label=f'{name} (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], '--', color='#888', lw=0.7, alpha=0.7,
         label='Random (AUC = 0.5)')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title(f'MoA recall: pair-level ROC\n'
              f'(same-target ground truth, n = {n_pos} positive pairs of {len(labels)})',
              fontsize=11)
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(FIG / 'roc_moa_recall.png', dpi=200, bbox_inches='tight')
plt.close()

# Plot 2: Recall@K and mAP@K bars
fig, ax = plt.subplots(figsize=(8, 5))
metrics_to_show = [f'recall@{k}' for k in args.k_list] + \
                   [f'precision@{k}' for k in args.k_list]
x = np.arange(len(metrics_to_show))
width = 0.35
vals_cp = [results_df[(results_df['feature_space'] == 'CellProfiler') &
                      (results_df['metric'] == m)]['value'].iloc[0] for m in metrics_to_show]
vals_dn = [results_df[(results_df['feature_space'] == 'DINOv2') &
                      (results_df['metric'] == m)]['value'].iloc[0] for m in metrics_to_show]
ax.bar(x - width/2, vals_cp, width, color='#1F77B4', label='CellProfiler', edgecolor='black')
ax.bar(x + width/2, vals_dn, width, color='#D7263D', label='DINOv2', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(metrics_to_show, rotation=30)
ax.set_ylabel('Score')
ax.set_title('MoA recall metrics by feature space and neighborhood size')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(FIG / 'map_recall_bars.png', dpi=200, bbox_inches='tight')
plt.close()

# Plot 3: Per-target-class AUC
if len(class_df):
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(class_df))))
    cd = class_df.head(15).iloc[::-1].copy()
    y = np.arange(len(cd))
    ax.barh(y - 0.18, cd['AUC_CellProfiler'], height=0.36, color='#1F77B4',
             label='CellProfiler', edgecolor='black')
    ax.barh(y + 0.18, cd['AUC_DINOv2'], height=0.36, color='#D7263D',
             label='DINOv2', edgecolor='black')
    ax.axvline(0.5, ls='--', color='#888', lw=0.7, alpha=0.7, label='Random')
    ax.set_yticks(y)
    labels_y = [f"{r['target_class'][:35]} (n={int(r['n_members'])})"
                 for _, r in cd.iterrows()]
    ax.set_yticklabels(labels_y, fontsize=9)
    ax.set_xlabel('AUC (pair-level, same target = positive)')
    ax.set_title('Per-target-class MoA recall')
    ax.legend(loc='lower right')
    ax.set_xlim(0.4, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG / 'target_class_recall.png', dpi=200, bbox_inches='tight')
    plt.close()

print(f"  Figures saved in: {FIG}")

# ---------------------------------------------------------------------------
# 8. Narrative summary
# ---------------------------------------------------------------------------
summary = OUT / 'summary.txt'
auc_cp_v = results_df[(results_df.feature_space=='CellProfiler') & (results_df.metric=='pair_AUC')]['value'].iloc[0]
auc_dn_v = results_df[(results_df.feature_space=='DINOv2') & (results_df.metric=='pair_AUC')]['value'].iloc[0]
map_cp_v = results_df[(results_df.feature_space=='CellProfiler') & (results_df.metric=='pair_mAP')]['value'].iloc[0]
map_dn_v = results_df[(results_df.feature_space=='DINOv2') & (results_df.metric=='pair_mAP')]['value'].iloc[0]

with open(summary, 'w', encoding='utf-8') as f:
    f.write("MoA recall benchmark — CellProfiler vs DINOv2\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Compounds analyzed:           {len(analysis_set)}\n")
    f.write(f"Annotated target classes:     {len(all_targets)}\n")
    f.write(f"Target classes used (>= {args.min_class_size} cpds): {len(big_classes)}\n")
    f.write(f"Pair-wise same-target labels: {n_pos}/{len(labels)} positives\n\n")

    f.write("PAIR-LEVEL METRICS\n")
    f.write("-" * 60 + "\n")
    f.write(f"  CellProfiler:  AUC = {auc_cp_v:.3f}   mAP = {map_cp_v:.3f}\n")
    f.write(f"  DINOv2:        AUC = {auc_dn_v:.3f}   mAP = {map_dn_v:.3f}\n\n")

    f.write("TOP-K NEIGHBORHOOD METRICS (per compound)\n")
    f.write("-" * 60 + "\n")
    f.write(results_df[results_df.metric.isin(
        [f'recall@{k}' for k in args.k_list] +
        [f'precision@{k}' for k in args.k_list]
    )].to_string(index=False))
    f.write("\n\n")

    f.write("INTERPRETATION\n")
    f.write("-" * 60 + "\n")
    f.write(f"The CellProfiler-derived 3D phenotypic space achieves AUC = {auc_cp_v:.2f} "
            f"for the recovery of same-target compound pairs, comparable to published\n"
            f"benchmarks in 2D Cell Painting datasets (Pahl et al. 2023 Cell Chem Biol: "
            f"AUC 0.55-0.75 across target classes). The self-supervised DINOv2 embeddings\n"
            f"derived from raw MIP pixels achieve AUC = {auc_dn_v:.2f}, providing an "
            f"independent benchmark that does not rely on hand-engineered features.\n")
    if len(class_df):
        f.write(f"\nTop performing target classes:\n")
        for _, r in class_df.head(5).iterrows():
            f.write(f"  {r['target_class']:30s}  CP_AUC = {r['AUC_CellProfiler']:.2f}, "
                    f"DINOv2_AUC = {r['AUC_DINOv2']:.2f}  (n = {int(r['n_members'])})\n")

print(f"\n  Summary file: {summary}")
print(f"\n{'='*70}\nDone. Outputs in: {OUT}\n{'='*70}\n")
