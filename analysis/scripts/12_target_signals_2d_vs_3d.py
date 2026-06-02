"""
12_target_signals_2d_vs_3d.py
==============================
Characterize the chemical-biological signal captured by 2D vs 3D Cell Painting
in HepG2 cells within the EU-OPENSCREEN Bioactive Compound Set.

For each modality (2D and 3D), this script identifies the top-N most active
compounds within the matched bioactive library and reports the targets, MoAs,
ChEMBL hierarchy and gene categories enriched in each group. The comparison
is matched at N = 25 (same as the 3D hit count) to enable direct contrast.

Inputs
------
results/medina_2d_vs_3d/per_compound.csv     (built by script 11)
data/annotated/cp3d_library_annotated.csv    (built by script 10)

Outputs (saved in results/medina_2d_vs_3d/target_signals/)
----------------------------------------------------------
top25_2D_compounds.csv         List of top-25 compounds by 2D active feature count
top25_3D_compounds.csv         List of top-25 3D hits (CP3D pipeline)
overlap_compounds.csv          Intersection of the two top-25 lists
target_enrichment.csv          Fisher exact test per target (top-25 2D vs library,
                               top-25 3D vs library)
moa_enrichment.csv             Same for MoA categories
summary.txt                    Narrative summary
figures/venn_top25_2d_vs_3d.png
figures/top_moa_bars.png       Most frequent MoAs in each top-25 list

Usage
-----
    python 12_target_signals_2d_vs_3d.py
    python 12_target_signals_2d_vs_3d.py --top_n 25        # change matched N
    python 12_target_signals_2d_vs_3d.py --top_metric RMS  # use 2D RMS instead of n_features_active
"""

import argparse
import sys
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

BASE = Path(r'C:\Users\Ianezi\Documents\CP3D')
ANALYSIS = BASE / 'analysis'
LIB_FILE = ANALYSIS / 'data' / 'annotated' / 'cp3d_library_annotated.csv'
PER_COMPOUND = ANALYSIS / 'results' / 'medina_2d_vs_3d' / 'per_compound.csv'
OUT = ANALYSIS / 'results' / 'medina_2d_vs_3d' / 'target_signals'
FIG = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--top_n', type=int, default=25,
                    help='Top-N compounds to compare in each modality (default 25)')
parser.add_argument('--top_metric', choices=['n_active', 'RMS'], default='n_active',
                    help='2D metric to define top-N (default: n_features_active)')
args = parser.parse_args()

print(f"\n{'='*70}")
print(f"=== Target/MoA signal comparison: top {args.top_n} 2D vs top {args.top_n} 3D ===")
print(f"{'='*70}\n")

# ---------------------------------------------------------------------------
# 1. Load inputs
# ---------------------------------------------------------------------------
print(f"{'-'*70}\n1. Loading inputs\n{'-'*70}")
if not PER_COMPOUND.exists() or not LIB_FILE.exists():
    print(f"  ERROR: missing inputs. Run scripts 10 and 11 first.")
    sys.exit(1)
df = pd.read_csv(PER_COMPOUND)
lib = pd.read_csv(LIB_FILE)
print(f"  per_compound dataset: {len(df)} compounds, {df.shape[1]} columns")
print(f"  annotated library:    {len(lib)} compounds, {lib.shape[1]} columns")

# Merge MoA / target info from library
anno_cols = ['EOS_id', 'Drug_name', 'MoA', 'Target_name', 'Gene_name', 'Target_type',
             'N_targets', 'ChEMBL_L1', 'ChEMBL_L2', 'ChEMBL_L3',
             'GTOPDB_L1', 'GTOPDB_L2']
anno_cols = [c for c in anno_cols if c in lib.columns]
df = df.drop(columns=[c for c in anno_cols if c in df.columns and c != 'EOS_id'])
df = df.merge(lib[anno_cols], on='EOS_id', how='left')
print(f"  Merged annotation columns: {[c for c in anno_cols if c != 'EOS_id']}")

# ---------------------------------------------------------------------------
# 2. Define top-N groups in 2D and 3D
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n2. Defining top-N compound sets\n{'-'*70}")
metric_2d = 'n_features_active' if args.top_metric == 'n_active' else 'activity_RMS_2D'

top_2d = df.nlargest(args.top_n, metric_2d).copy().sort_values(metric_2d, ascending=False)
top_3d = df[df['is_hit_3D']].copy().sort_values('activity_RMS_3D', ascending=False)

print(f"  Top {args.top_n} in 2D (by {metric_2d}): {len(top_2d)} compounds")
print(f"  Top 3D (CP3D pipeline hits):            {len(top_3d)} compounds")

eos_2d = set(top_2d['EOS_id'])
eos_3d = set(top_3d['EOS_id'])
overlap = eos_2d & eos_3d
only_2d = eos_2d - eos_3d
only_3d = eos_3d - eos_2d

print(f"\n  Compound-level partition:")
print(f"    Top-{args.top_n} 2D only:  {len(only_2d)}")
print(f"    Top-{args.top_n} 3D only:  {len(only_3d)}")
print(f"    Shared in both:        {len(overlap)}")

# Save compound-level lists
display_cols = ['EOS_id', 'Drug_name', 'MoA', 'Target_name', 'Gene_name',
                'n_features_active', 'activity_RMS_2D', 'activity_RMS_3D',
                'rank_2D_n_active_pct', 'rank_3D_RMS_pct']
display_cols = [c for c in display_cols if c in df.columns]

top_2d[display_cols].to_csv(OUT / f'top{args.top_n}_2D_compounds.csv', index=False)
top_3d[display_cols].to_csv(OUT / f'top{args.top_n}_3D_compounds.csv', index=False)

overlap_df = top_2d[top_2d['EOS_id'].isin(overlap)][display_cols]
overlap_df.to_csv(OUT / 'overlap_compounds.csv', index=False)

# ---------------------------------------------------------------------------
# 3. Print compound lists side by side
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n3. Compound lists by modality\n{'-'*70}")

print(f"\n  TOP {args.top_n} in 2D (by active feature count):")
print(f"  {'#':>3}  {'EOS_id':<11} {'Drug_name':<28} {'MoA':<30} n_active")
for i, (_, r) in enumerate(top_2d.iterrows(), 1):
    drug = (str(r['Drug_name']) if pd.notna(r['Drug_name']) else r['EOS_id'])[:27]
    moa = (str(r['MoA']) if pd.notna(r['MoA']) else 'NA')[:29]
    flag = '*' if r['EOS_id'] in overlap else ' '
    print(f"  {i:>3}{flag} {r['EOS_id']:<11} {drug:<28} {moa:<30} {int(r['n_features_active'])}")

print(f"\n  TOP {len(top_3d)} 3D hits (CP3D pipeline):")
print(f"  {'#':>3}  {'EOS_id':<11} {'Drug_name':<28} {'MoA':<30} RMS_3D")
for i, (_, r) in enumerate(top_3d.iterrows(), 1):
    drug = (str(r['Drug_name']) if pd.notna(r['Drug_name']) else r['EOS_id'])[:27]
    moa = (str(r['MoA']) if pd.notna(r['MoA']) else 'NA')[:29]
    flag = '*' if r['EOS_id'] in overlap else ' '
    print(f"  {i:>3}{flag} {r['EOS_id']:<11} {drug:<28} {moa:<30} {r['activity_RMS_3D']:.2f}")

print(f"\n  (* = compound shared between both top-{args.top_n} lists)")

# ---------------------------------------------------------------------------
# 4. Target-level enrichment (Fisher exact)
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n4. Target / MoA / Gene enrichment\n{'-'*70}")

def expand_multi_target(series):
    """Expand comma- or semicolon-separated target strings into a list of (idx, target)."""
    pairs = []
    for idx, val in series.items():
        if pd.isna(val):
            continue
        for tgt in str(val).replace(';', ',').split(','):
            t = tgt.strip()
            if t:
                pairs.append((idx, t))
    return pairs

def enrichment_table(level_col, label):
    """Compute Fisher exact enrichment for each unique value of `level_col`
    in top-2D and top-3D vs the full analytical library."""
    pairs = expand_multi_target(df[level_col])
    if not pairs:
        return pd.DataFrame()
    long_df = pd.DataFrame(pairs, columns=['orig_idx', label])
    long_df = long_df.merge(df[['EOS_id', 'is_hit_3D']].reset_index().rename(
        columns={'index': 'orig_idx'}), on='orig_idx')
    long_df['in_top_2D'] = long_df['EOS_id'].isin(eos_2d)
    long_df['in_top_3D'] = long_df['EOS_id'].isin(eos_3d)

    rows = []
    counts = long_df[label].value_counts()
    for value, n_total in counts.items():
        if n_total < 2:
            continue
        # Use compound-level counts (deduplicate by EOS within each top set)
        cs = long_df[long_df[label] == value]
        n_2d = cs['in_top_2D'].sum()
        n_3d = cs['in_top_3D'].sum()
        # Fisher: in top vs not in top, for compounds annotated to this category
        # vs all other compounds in the library
        for top_label, top_eos, n_top in [('2D', eos_2d, n_2d),
                                            ('3D', eos_3d, n_3d)]:
            # Build 2x2: rows = annotated to this category (yes/no);
            #            cols = in top-N (yes/no)
            n_annot = n_total
            n_lib = len(df)
            a = n_top                             # in_category AND in_top
            b = n_annot - n_top                   # in_category AND NOT in_top
            c = len(top_eos) - n_top              # NOT in_category AND in_top
            d = n_lib - n_annot - c               # NOT in_category AND NOT in_top
            if min(a, b, c, d) < 0:
                continue
            try:
                odds, p = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
            except Exception:
                odds, p = np.nan, np.nan
            rows.append({
                'category_level': label,
                'category_value': value,
                'modality': top_label,
                'n_in_top': int(n_top),
                'n_total_library': int(n_total),
                'fisher_OR': odds,
                'fisher_p_one_sided': p,
            })
    out = pd.DataFrame(rows).sort_values(['modality', 'fisher_p_one_sided'])
    # FDR correction per modality
    if len(out):
        for mod in out['modality'].unique():
            mask = out['modality'] == mod
            pvals = out.loc[mask, 'fisher_p_one_sided'].values
            sorted_idx = np.argsort(pvals)
            n = len(pvals)
            fdr = np.zeros(n)
            for i in range(n):
                fdr[sorted_idx[i]] = pvals[sorted_idx[i]] * n / (i + 1)
            fdr = np.minimum.accumulate(fdr[sorted_idx][::-1])[::-1]
            out.loc[mask, 'fisher_FDR_BH'] = fdr
    return out

# Target_name level enrichment
target_enrich = enrichment_table('Target_name', 'Target_name')
if len(target_enrich):
    target_enrich.to_csv(OUT / 'target_enrichment.csv', index=False)
    sig = target_enrich[target_enrich['fisher_p_one_sided'] < 0.05]
    print(f"\n  Targets with raw p < 0.05 (n = {len(sig)}):")
    if len(sig):
        print(sig[['modality', 'category_value', 'n_in_top', 'n_total_library',
                   'fisher_OR', 'fisher_p_one_sided']].to_string(index=False))

# MoA level enrichment
moa_enrich = enrichment_table('MoA', 'MoA')
if len(moa_enrich):
    moa_enrich.to_csv(OUT / 'moa_enrichment.csv', index=False)
    sig = moa_enrich[moa_enrich['fisher_p_one_sided'] < 0.05]
    print(f"\n  MoA categories with raw p < 0.05 (n = {len(sig)}):")
    if len(sig):
        print(sig[['modality', 'category_value', 'n_in_top', 'n_total_library',
                   'fisher_OR', 'fisher_p_one_sided']].to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Compact MoA category counts side by side
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n5. MoA frequencies in each top-N list\n{'-'*70}")
def top_moa_counts(subset, n=10):
    pairs = expand_multi_target(subset['MoA'])
    if not pairs:
        return Counter()
    long = pd.DataFrame(pairs, columns=['orig_idx', 'MoA'])
    return Counter(long['MoA'])

moa_2d = top_moa_counts(top_2d)
moa_3d = top_moa_counts(top_3d)

all_moas = set(moa_2d) | set(moa_3d)
moa_summary = pd.DataFrame([
    {'MoA': m,
     f'count_top{args.top_n}_2D': moa_2d.get(m, 0),
     f'count_top{args.top_n}_3D': moa_3d.get(m, 0)}
    for m in all_moas
]).sort_values(f'count_top{args.top_n}_3D', ascending=False)
moa_summary.to_csv(OUT / 'moa_counts_side_by_side.csv', index=False)
print(moa_summary.head(15).to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n6. Figures\n{'-'*70}")

# Venn diagram
try:
    from matplotlib_venn import venn2
    fig, ax = plt.subplots(figsize=(7, 6))
    venn2([eos_2d, eos_3d], (f'Top {args.top_n}\n2D MEDINA',
                              f'Top {len(eos_3d)}\n3D MEDINA'), ax=ax)
    ax.set_title(f'Overlap of top compounds across modalities '
                 f'(within {len(df)} bioactives)')
    plt.tight_layout()
    plt.savefig(FIG / 'venn_top_2d_vs_3d.png', dpi=150, bbox_inches='tight')
    plt.close()
except ImportError:
    # Fallback: simple bar chart of partition sizes
    fig, ax = plt.subplots(figsize=(7, 5))
    cats = [f'2D-only', 'Shared', f'3D-only']
    counts = [len(only_2d), len(overlap), len(only_3d)]
    ax.bar(cats, counts, color=['#377EB8', '#984EA3', '#E41A1C'], edgecolor='black')
    for i, v in enumerate(counts):
        ax.text(i, v + 0.3, str(v), ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of compounds')
    ax.set_title(f'Partition of top-{args.top_n} compounds across modalities')
    plt.tight_layout()
    plt.savefig(FIG / 'venn_top_2d_vs_3d.png', dpi=150, bbox_inches='tight')
    plt.close()

# Top MoA bars side by side
top_moas = (pd.DataFrame({'MoA': list(all_moas),
                          'count_2D': [moa_2d.get(m, 0) for m in all_moas],
                          'count_3D': [moa_3d.get(m, 0) for m in all_moas]})
            .assign(total=lambda d: d['count_2D'] + d['count_3D'])
            .sort_values('total', ascending=False).head(15))
fig, ax = plt.subplots(figsize=(10, 8))
y = np.arange(len(top_moas))
ax.barh(y - 0.18, top_moas['count_2D'], height=0.36, color='#377EB8',
        label=f'Top {args.top_n} 2D')
ax.barh(y + 0.18, top_moas['count_3D'], height=0.36, color='#E41A1C',
        label=f'Top {args.top_n} 3D')
ax.set_yticks(y)
ax.set_yticklabels([str(m)[:50] for m in top_moas['MoA']], fontsize=9)
ax.set_xlabel('Number of compounds in top list')
ax.set_title(f'Most frequent MoAs in 2D vs 3D top-{args.top_n}')
ax.legend()
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIG / 'top_moa_bars.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"  Figures saved in: {FIG}")

# ---------------------------------------------------------------------------
# 7. Narrative summary
# ---------------------------------------------------------------------------
summary = OUT / 'summary.txt'
with open(summary, 'w', encoding='utf-8') as f:
    f.write("Target / MoA signals captured by 2D vs 3D Cell Painting\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Top-N comparison set at N = {args.top_n} (matched to the 25 robust 3D hits).\n")
    f.write(f"2D metric: {metric_2d}\n")
    f.write(f"Library size: {len(df)} bioactive compounds.\n\n")

    f.write("COMPOUND-LEVEL PARTITION\n")
    f.write("------------------------\n")
    f.write(f"  Top {args.top_n} 2D only:  {len(only_2d)}\n")
    f.write(f"  Top {args.top_n} 3D only:  {len(only_3d)}\n")
    f.write(f"  Shared in both:        {len(overlap)}\n\n")

    if overlap:
        f.write("  Compounds shared between top-2D and top-3D:\n")
        for _, r in overlap_df.iterrows():
            drug = (str(r['Drug_name']) if pd.notna(r['Drug_name']) else r['EOS_id'])
            moa = str(r['MoA']) if pd.notna(r['MoA']) else 'NA'
            f.write(f"    {r['EOS_id']:<12} {drug:<28} | MoA: {moa}\n")
        f.write("\n")

    f.write(f"TOP {args.top_n} 2D BY ACTIVE FEATURE COUNT\n")
    f.write("-" * 60 + "\n")
    for _, r in top_2d.iterrows():
        drug = (str(r['Drug_name']) if pd.notna(r['Drug_name']) else r['EOS_id'])
        moa = str(r['MoA']) if pd.notna(r['MoA']) else 'NA'
        f.write(f"  {r['EOS_id']:<12} {drug:<28} | n_active = {int(r['n_features_active'])} "
                f"| MoA: {moa}\n")

    f.write(f"\nTOP {len(top_3d)} 3D HITS (CP3D pipeline)\n")
    f.write("-" * 60 + "\n")
    for _, r in top_3d.iterrows():
        drug = (str(r['Drug_name']) if pd.notna(r['Drug_name']) else r['EOS_id'])
        moa = str(r['MoA']) if pd.notna(r['MoA']) else 'NA'
        f.write(f"  {r['EOS_id']:<12} {drug:<28} | RMS_3D = {r['activity_RMS_3D']:.2f} "
                f"| MoA: {moa}\n")

    f.write(f"\nMoA FREQUENCIES (top 15 by combined count)\n")
    f.write("-" * 60 + "\n")
    f.write(top_moas.to_string(index=False))
    f.write("\n")

print(f"\n  Summary file: {summary}")
print(f"\n{'='*70}\nAnalysis complete. Outputs in: {OUT}\n{'='*70}\n")
