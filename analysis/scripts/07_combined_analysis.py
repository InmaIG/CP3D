"""
07_combined_analysis.py — Cross-plate combined analysis (C2386 + C2387 + C2388)
================================================================================
Combina los confirmed hits + lost-in-3D de las 3 placas en un solo dataset
para hacer enriquecimiento estadisticamente potente con ~1700 compuestos.

Tambien hace analisis especifico de promiscuidad (n_targets) para ver si los
hits 3D son mas/menos promiscuos que los lost-in-3D.

Outputs en results\\enrichment\\combined\\:
- combined_compounds_annotated.csv  (todos los compuestos con hit/MoA)
- combined_moa_enrichment.csv        (Fisher por MoA)
- combined_target_enrichment.csv     (Fisher por target)
- combined_gene_enrichment.csv       (Fisher por gen)
- combined_chembl_enrichment.csv     (todos los niveles ChEMBL)
- combined_gtopdb_enrichment.csv     (todos los niveles GTOPDB)
- combined_promiscuity_analysis.csv  (analisis especifico de n_targets)
- combined_promiscuity_plot.png      (plot promiscuidad)
- combined_top_enriched.png          (visual)
- combined_summary.txt               (resumen general)

Uso:
    python 07_combined_analysis.py
    python 07_combined_analysis.py --plates C2386 C2387 C2388
"""

import argparse
import sys
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore')

BASE = Path(r'C:\Users\Ianezi\Documents\CP3D\analysis')
PROC_DIR = BASE / 'data' / 'processed'
OUT_DIR = BASE / 'results' / 'enrichment' / 'combined'
OUT_DIR.mkdir(parents=True, exist_ok=True)

LIBRARY_FILE = Path(r'C:\Users\Ianezi\Documents\CP3D\EOS_compounds_MoA.csv')

parser = argparse.ArgumentParser()
parser.add_argument('--plates', nargs='+', default=['C2386', 'C2387', 'C2388'])
parser.add_argument('--fdr_threshold', type=float, default=0.25)
parser.add_argument('--min_replicates', type=int, default=3,
                    help='Minimum N_replicates to consider a hit (default: 3)')
args = parser.parse_args()
plates = args.plates
min_replicates = args.min_replicates

print(f"\n{'='*70}\n=== COMBINED ANALYSIS: {plates} ===\n{'='*70}")
print(f"Library: {LIBRARY_FILE}")
print(f"FDR threshold: {args.fdr_threshold}")
print(f"Min replicates filter: {min_replicates}")

# ---------------- Load library ----------------
try:
    library = pd.read_csv(LIBRARY_FILE, encoding='latin-1')
except UnicodeDecodeError:
    library = pd.read_csv(LIBRARY_FILE, encoding='utf-8', errors='replace')

eos_col = 'EOS' if 'EOS' in library.columns else 'EOS_id'
print(f"\nLibrary: {len(library)} compounds, {len(library.columns)} cols")

# ---------------- Combine hits across plates ----------------
print(f"\n{'-'*70}\nLoad hits from each plate\n{'-'*70}")

all_compounds_annotated = []
hit_summary = {}

for plate in plates:
    hits_path = PROC_DIR / f'confirmed_hits_{plate}.csv'
    if not hits_path.exists():
        print(f"  WARNING: No hits file for {plate}, skipping")
        continue
    hits = pd.read_csv(hits_path)
    
    # Categorize WITH min_replicates filter
    def categorize(row):
        n_reps = row.get('Metadata_n_replicates', 0)
        
        # First identify candidate
        if row['is_confirmed_hit']:
            candidate = 'confirmed_hit'
        elif row['is_active_top5pct'] and not row['is_reproducible']:
            if n_reps == 1:
                candidate = 'highly_toxic'
            else:
                candidate = 'active_only'
        elif row['is_reproducible'] and not row['is_active_top5pct']:
            candidate = 'reproducible_only'
        else:
            candidate = 'inactive'
        
        # Apply min_replicates filter for hits
        if candidate in ['confirmed_hit', 'highly_toxic']:
            if n_reps < min_replicates:
                return 'excluded_low_reps'
        return candidate
    
    hits['Hit_category'] = hits.apply(categorize, axis=1)
    
    # Define 3D hits = confirmed + highly toxic (with N >= min_replicates)
    hits['Is_3D_hit'] = hits['Hit_category'].isin(['confirmed_hit', 'highly_toxic'])
    hits['Plate'] = plate
    
    n_hits = hits['Is_3D_hit'].sum()
    n_excluded = (hits['Hit_category'] == 'excluded_low_reps').sum()
    n_total = len(hits)
    print(f"  {plate}: {n_total} compounds, {n_hits} 3D hits ({100*n_hits/n_total:.1f}%)" +
          (f", {n_excluded} excluded (N<{min_replicates})" if n_excluded > 0 else ""))
    hit_summary[plate] = {'n_total': n_total, 'n_hits': n_hits, 'n_excluded': n_excluded}
    
    all_compounds_annotated.append(hits[['Metadata_Compound', 'Plate', 'Hit_category', 
                                          'Is_3D_hit', 'Metadata_Activity',
                                          'Metadata_Replicate_corr', 'AreaShape_Area_consensus']])

if not all_compounds_annotated:
    print("\nERROR: No hits files found")
    sys.exit(1)

combined = pd.concat(all_compounds_annotated, ignore_index=True)
print(f"\nCombined dataset: {len(combined)} compounds (across {len(plates)} plates)")

# ---------------- Merge with library ----------------
print(f"\n{'-'*70}\nMerge with library MoA info\n{'-'*70}")
# Drop 'Plate' from library to avoid _x/_y suffixes (we have our own Plate column)
library_for_merge = library.drop(columns=['Plate'], errors='ignore')
combined = combined.merge(library_for_merge, left_on='Metadata_Compound', right_on=eos_col, how='left')
n_with_info = combined['EUopen_name'].notna().sum()
n_without = combined['EUopen_name'].isna().sum()
print(f"  With library info: {n_with_info}")
print(f"  Without library info: {n_without}")

# Save annotated table
out_annotated = OUT_DIR / 'combined_compounds_annotated.csv'
combined.to_csv(out_annotated, index=False)
print(f"  Saved: {out_annotated}")

# ---------------- Define 3D hits vs Lost ----------------
hits_3d = combined[combined['Is_3D_hit']].copy()
# Exclude 'excluded_low_reps' from lost — they are not lost, just unvalidated
lost_3d = combined[(~combined['Is_3D_hit']) & 
                    (combined['Hit_category'] != 'excluded_low_reps')].copy()
n_excluded_total = (combined['Hit_category'] == 'excluded_low_reps').sum()

print(f"\n{'-'*70}\nGroups for enrichment\n{'-'*70}")
print(f"  3D hits (confirmed + highly_toxic, N>={min_replicates}): {len(hits_3d)}")
print(f"    confirmed_hit: {(hits_3d['Hit_category']=='confirmed_hit').sum()}")
print(f"    highly_toxic:  {(hits_3d['Hit_category']=='highly_toxic').sum()}")
if n_excluded_total > 0:
    print(f"  Excluded (low replicates, removed from analysis): {n_excluded_total}")
print(f"  Lost in 3D: {len(lost_3d)}")

# ---------------- Fisher exact test helper ----------------
def fisher_enrichment(items, df_hits, df_lost, item_col):
    """Fisher exact test for enrichment of each item among hits vs lost."""
    results = []
    n_hits_total = len(df_hits)
    n_lost_total = len(df_lost)
    
    # Expand multi-value items (semicolon-separated)
    def get_items(s):
        if pd.isna(s) or not s: return []
        return [x.strip() for x in str(s).split(';') if x.strip()]
    
    all_items = set()
    for s in df_hits[item_col].dropna():
        all_items.update(get_items(s))
    for s in df_lost[item_col].dropna():
        all_items.update(get_items(s))
    
    for item in all_items:
        hits_with = sum(1 for s in df_hits[item_col].dropna() if item in get_items(s))
        lost_with = sum(1 for s in df_lost[item_col].dropna() if item in get_items(s))
        hits_without = n_hits_total - hits_with
        lost_without = n_lost_total - lost_with
        
        # Fisher exact: contingency table [[hits_with, hits_without], [lost_with, lost_without]]
        if hits_with == 0 and lost_with == 0:
            continue
        try:
            odds, pval_one = stats.fisher_exact(
                [[hits_with, hits_without], [lost_with, lost_without]],
                alternative='greater')
            _, pval_two = stats.fisher_exact(
                [[hits_with, hits_without], [lost_with, lost_without]])
        except:
            continue
        
        results.append({
            'item': item,
            'hits_count': hits_with,
            'hits_pct': 100 * hits_with / n_hits_total if n_hits_total > 0 else 0,
            'lost_count': lost_with,
            'lost_pct': 100 * lost_with / n_lost_total if n_lost_total > 0 else 0,
            'odds_ratio': odds,
            'p_value': pval_one,
            'p_value_two_sided': pval_two,
        })
    
    df = pd.DataFrame(results)
    if len(df) > 0:
        # FDR (Benjamini-Hochberg)
        df = df.sort_values('p_value').reset_index(drop=True)
        n = len(df)
        df['fdr'] = (df['p_value'].values * n / np.arange(1, n+1)).clip(0, 1)
        # Make FDR monotonic
        df['fdr'] = df['fdr'][::-1].cummin()[::-1]
    return df

# ---------------- Enrichment: MoA ----------------
print(f"\n{'-'*70}\nEnrichment: MoA\n{'-'*70}")
moa_df = fisher_enrichment(None, hits_3d, lost_3d, 'EUopen_moa')
moa_df = moa_df.sort_values('p_value')
print(f"  Tested {len(moa_df)} MoAs")
print(f"\n  Top 10 by p-value:")
print(moa_df.head(10).to_string(index=False))
moa_df.to_csv(OUT_DIR / 'combined_moa_enrichment.csv', index=False)

# ---------------- Enrichment: target_name ----------------
print(f"\n{'-'*70}\nEnrichment: target_name\n{'-'*70}")
target_df = fisher_enrichment(None, hits_3d, lost_3d, 'EUopen_target_name')
target_df = target_df.sort_values('p_value')
print(f"  Tested {len(target_df)} targets")
print(f"\n  Top 15 by p-value:")
print(target_df.head(15).to_string(index=False))
target_df.to_csv(OUT_DIR / 'combined_target_enrichment.csv', index=False)

# Significant after FDR
sig_targets = target_df[target_df['fdr'] < args.fdr_threshold]
if len(sig_targets) > 0:
    print(f"\n  *** Targets significant at FDR < {args.fdr_threshold}: {len(sig_targets)} ***")

# ---------------- Enrichment: gene_name ----------------
print(f"\n{'-'*70}\nEnrichment: gene_name\n{'-'*70}")
gene_df = fisher_enrichment(None, hits_3d, lost_3d, 'EUopen_gene_name')
gene_df = gene_df.sort_values('p_value')
print(f"  Tested {len(gene_df)} genes")
print(f"\n  Top 15 enriched genes:")
print(gene_df.head(15).to_string(index=False))
gene_df.to_csv(OUT_DIR / 'combined_gene_enrichment.csv', index=False)

sig_genes = gene_df[gene_df['fdr'] < args.fdr_threshold]
if len(sig_genes) > 0:
    print(f"\n  *** Genes significant at FDR < {args.fdr_threshold}: {len(sig_genes)} ***")

# ---------------- Enrichment: ChEMBL classification ----------------
print(f"\n{'-'*70}\nEnrichment: ChEMBL classification levels\n{'-'*70}")
chembl_all = []
for level in [1, 2, 3, 4]:
    col = f'EUopen_ChEMBL [LEVEL {level}]'
    if col not in hits_3d.columns:
        continue
    df = fisher_enrichment(None, hits_3d, lost_3d, col)
    if len(df) == 0:
        continue
    df['chembl_level'] = level
    df = df.sort_values('p_value')
    print(f"\n  Level {level}: {len(df)} tested. Top 5:")
    print(df.head(5).to_string(index=False))
    chembl_all.append(df)

if chembl_all:
    chembl_combined = pd.concat(chembl_all, ignore_index=True)
    chembl_combined.to_csv(OUT_DIR / 'combined_chembl_enrichment.csv', index=False)
    print(f"\n  Saved: combined_chembl_enrichment.csv")

# ---------------- Enrichment: GTOPDB classification ----------------
print(f"\n{'-'*70}\nEnrichment: GTOPDB classification\n{'-'*70}")
gtopdb_all = []
for level in [1, 2, 3]:
    col = f'EUopen_GTOPDB [LEVEL {level}]'
    if col not in hits_3d.columns:
        continue
    df = fisher_enrichment(None, hits_3d, lost_3d, col)
    if len(df) == 0:
        continue
    df['gtopdb_level'] = level
    df = df.sort_values('p_value')
    print(f"\n  Level {level}: {len(df)} tested. Top 5:")
    print(df.head(5).to_string(index=False))
    gtopdb_all.append(df)

if gtopdb_all:
    gtopdb_combined = pd.concat(gtopdb_all, ignore_index=True)
    gtopdb_combined.to_csv(OUT_DIR / 'combined_gtopdb_enrichment.csv', index=False)

# ---------------- PROMISCUITY ANALYSIS ----------------
print(f"\n{'='*70}\n=== PROMISCUITY ANALYSIS ===\n{'='*70}")

n_targets_col = 'EUopen_no. targets'
if n_targets_col not in combined.columns:
    print("WARNING: n_targets column not found, skipping")
else:
    hits_n = pd.to_numeric(hits_3d[n_targets_col], errors='coerce').dropna()
    lost_n = pd.to_numeric(lost_3d[n_targets_col], errors='coerce').dropna()
    
    print(f"\n3D hits (n={len(hits_n)}):")
    print(f"  Median: {hits_n.median():.1f}")
    print(f"  Mean:   {hits_n.mean():.1f}")
    print(f"  P25-P75: {hits_n.quantile(0.25):.1f} - {hits_n.quantile(0.75):.1f}")
    print(f"  Promiscuous (>20 targets): {(hits_n > 20).sum()} ({100*(hits_n > 20).sum()/len(hits_n):.1f}%)")
    print(f"  Selective (<=5 targets):   {(hits_n <= 5).sum()} ({100*(hits_n <= 5).sum()/len(hits_n):.1f}%)")
    
    print(f"\nLost in 3D (n={len(lost_n)}):")
    print(f"  Median: {lost_n.median():.1f}")
    print(f"  Mean:   {lost_n.mean():.1f}")
    print(f"  P25-P75: {lost_n.quantile(0.25):.1f} - {lost_n.quantile(0.75):.1f}")
    print(f"  Promiscuous (>20 targets): {(lost_n > 20).sum()} ({100*(lost_n > 20).sum()/len(lost_n):.1f}%)")
    print(f"  Selective (<=5 targets):   {(lost_n <= 5).sum()} ({100*(lost_n <= 5).sum()/len(lost_n):.1f}%)")
    
    # Mann-Whitney
    u, p_mw = stats.mannwhitneyu(hits_n, lost_n, alternative='two-sided')
    print(f"\nMann-Whitney U test:")
    print(f"  p-value: {p_mw:.4f}")
    if p_mw < 0.05:
        if hits_n.median() > lost_n.median():
            print(f"  ** Hits 3D are MORE promiscuous than Lost in 3D **")
        else:
            print(f"  ** Hits 3D are LESS promiscuous than Lost in 3D **")
    else:
        print(f"  No significant difference in promiscuity")
    
    # Fisher exact: promiscuous (>20) vs selective (<=20)
    a = (hits_n > 20).sum(); b = (hits_n <= 20).sum()
    c = (lost_n > 20).sum(); d = (lost_n <= 20).sum()
    odds, p_fisher = stats.fisher_exact([[a,b],[c,d]])
    print(f"\nFisher exact (promiscuous >20 targets):")
    print(f"  Hits: {a}/{a+b} ({100*a/(a+b):.1f}%) vs Lost: {c}/{c+d} ({100*c/(c+d):.1f}%)")
    print(f"  OR={odds:.2f}, p={p_fisher:.4f}")
    
    # Per-plate analysis
    print(f"\nPromiscuity by plate:")
    promiscuity_results = []
    for plate in plates:
        sub = combined[combined['Plate']==plate]
        h = pd.to_numeric(sub[sub['Is_3D_hit']][n_targets_col], errors='coerce').dropna()
        # Exclude excluded_low_reps from lost
        l_sub = sub[(~sub['Is_3D_hit']) & (sub['Hit_category'] != 'excluded_low_reps')]
        l = pd.to_numeric(l_sub[n_targets_col], errors='coerce').dropna()
        if len(h) > 0 and len(l) > 0:
            try:
                _, p = stats.mannwhitneyu(h, l, alternative='two-sided')
            except:
                p = np.nan
        else:
            p = np.nan
        result = {
            'plate': plate,
            'hits_n': len(h),
            'hits_median_targets': h.median() if len(h) > 0 else np.nan,
            'lost_n': len(l),
            'lost_median_targets': l.median() if len(l) > 0 else np.nan,
            'mannwhitney_p': p,
        }
        promiscuity_results.append(result)
        print(f"  {plate}: hits median={h.median():.1f} (n={len(h)}) vs lost median={l.median():.1f} (n={len(l)}), p={p:.4f}" 
              if not pd.isna(p) else f"  {plate}: insufficient data")
    
    promiscuity_df = pd.DataFrame(promiscuity_results)
    promiscuity_df.to_csv(OUT_DIR / 'combined_promiscuity_analysis.csv', index=False)
    
    # Plot promiscuity
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Box plot global
    ax = axes[0]
    bp = ax.boxplot([hits_n, lost_n], tick_labels=[f'3D hits\n(n={len(hits_n)})', f'Lost in 3D\n(n={len(lost_n)})'],
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], ['#E74C3C', '#3498DB']):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_ylabel('Number of targets')
    ax.set_title(f'Promiscuity (combined)\nMann-Whitney p={p_mw:.4f}')
    ax.grid(alpha=0.3, axis='y')
    
    # Histogram
    ax = axes[1]
    bins = np.linspace(0, max(hits_n.max(), lost_n.max()), 30)
    ax.hist(lost_n, bins=bins, color='#3498DB', alpha=0.5, label=f'Lost in 3D (n={len(lost_n)})', density=True)
    ax.hist(hits_n, bins=bins, color='#E74C3C', alpha=0.5, label=f'3D hits (n={len(hits_n)})', density=True)
    ax.axvline(hits_n.median(), color='#C0392B', linestyle='--', label=f'Hits median={hits_n.median():.1f}')
    ax.axvline(lost_n.median(), color='#2980B9', linestyle='--', label=f'Lost median={lost_n.median():.1f}')
    ax.set_xlabel('Number of targets')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of n_targets')
    ax.legend(fontsize=9)
    ax.set_xlim(0, hits_n.quantile(0.99) if hits_n.quantile(0.99) > 0 else 100)
    ax.grid(alpha=0.3)
    
    # Per plate
    ax = axes[2]
    x_pos = np.arange(len(plates))
    width = 0.35
    hits_medians = [r['hits_median_targets'] for r in promiscuity_results]
    lost_medians = [r['lost_median_targets'] for r in promiscuity_results]
    ax.bar(x_pos - width/2, hits_medians, width, label='3D hits', color='#E74C3C', alpha=0.7)
    ax.bar(x_pos + width/2, lost_medians, width, label='Lost in 3D', color='#3498DB', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(plates)
    ax.set_ylabel('Median n_targets')
    ax.set_title('Promiscuity by plate')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle('Promiscuity analysis: 3D hits vs Lost in 3D', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plot_path = OUT_DIR / 'combined_promiscuity_plot.png'
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    print(f"\n  Plot saved: {plot_path}")
    plt.close()

# ---------------- Top enriched plot ----------------
print(f"\n{'-'*70}\nTop enriched plot\n{'-'*70}")

# Combine top from MoA, target, gene, ChEMBL L2
top_items = []
for df, label in [(moa_df, 'MoA'), (target_df, 'Target'), (gene_df, 'Gene')]:
    if len(df) > 0:
        for _, row in df.head(5).iterrows():
            if row['p_value'] < 0.1:
                top_items.append({
                    'item': row['item'][:40],
                    'category': label,
                    'odds_ratio': row['odds_ratio'],
                    'p_value': row['p_value'],
                    'fdr': row['fdr'],
                    'hits_count': row['hits_count'],
                    'lost_count': row['lost_count'],
                })

if top_items:
    top_df = pd.DataFrame(top_items).sort_values('p_value').head(20)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors_by_cat = {'MoA': '#E74C3C', 'Target': '#3498DB', 'Gene': '#27AE60'}
    bar_colors = [colors_by_cat.get(c, '#95A5A6') for c in top_df['category']]
    
    y_pos = np.arange(len(top_df))
    or_capped = top_df['odds_ratio'].replace([np.inf, -np.inf], np.nan).fillna(top_df['odds_ratio'].replace(np.inf, 100).max() * 1.2)
    
    ax.barh(y_pos, or_capped, color=bar_colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['category']}: {row['item']}" for _, row in top_df.iterrows()], fontsize=9)
    ax.set_xlabel('Odds ratio (3D hits / Lost in 3D)')
    ax.set_title(f'Top enriched items in 3D hits ({len(plates)} plates combined)')
    ax.invert_yaxis()
    
    # Add p-value labels
    for i, (_, row) in enumerate(top_df.iterrows()):
        x = or_capped.iloc[i]
        label = f"  p={row['p_value']:.1e} (n={row['hits_count']:.0f})"
        ax.text(x, i, label, va='center', fontsize=8)
    
    from matplotlib.patches import Patch
    legend = [Patch(color=c, label=cat) for cat, c in colors_by_cat.items()]
    ax.legend(handles=legend, loc='lower right')
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'combined_top_enriched.png', dpi=120, bbox_inches='tight')
    print(f"  Saved: combined_top_enriched.png")
    plt.close()

# ---------------- Summary ----------------
summary_path = OUT_DIR / 'combined_summary.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(f"COMBINED ANALYSIS — Cross-plate enrichment\n")
    f.write(f"="*70 + "\n\n")
    f.write(f"Plates: {plates}\n\n")
    f.write(f"Compounds:\n")
    for plate, summ in hit_summary.items():
        f.write(f"  {plate}: {summ['n_total']} compounds, {summ['n_hits']} 3D hits\n")
    f.write(f"\nTotal: {len(combined)} compounds, {len(hits_3d)} 3D hits, {len(lost_3d)} lost\n\n")
    f.write(f"Significant enrichments (FDR < {args.fdr_threshold}):\n\n")
    
    for label, df in [('MoA', moa_df), ('Targets', target_df), ('Genes', gene_df)]:
        sig = df[df['fdr'] < args.fdr_threshold] if len(df) > 0 else pd.DataFrame()
        f.write(f"{label}: {len(sig)} significant\n")
        for _, row in sig.head(15).iterrows():
            f.write(f"  {row['item']}: hits={row['hits_count']:.0f} ({row['hits_pct']:.1f}%), "
                    f"lost={row['lost_count']:.0f} ({row['lost_pct']:.1f}%), "
                    f"OR={row['odds_ratio']:.2f}, p={row['p_value']:.4f}, FDR={row['fdr']:.3f}\n")
        f.write("\n")
    
    if n_targets_col in combined.columns:
        f.write(f"Promiscuity analysis:\n")
        f.write(f"  3D hits median n_targets: {hits_n.median():.1f}\n")
        f.write(f"  Lost in 3D median n_targets: {lost_n.median():.1f}\n")
        f.write(f"  Mann-Whitney p: {p_mw:.4f}\n")

print(f"\n  Summary saved: {summary_path}")

print(f"\n{'='*70}\n=== DONE ===\n{'='*70}")
print(f"\nKey outputs in {OUT_DIR}:")
print(f"  - combined_compounds_annotated.csv")
print(f"  - combined_moa_enrichment.csv")
print(f"  - combined_target_enrichment.csv")
print(f"  - combined_gene_enrichment.csv")
print(f"  - combined_chembl_enrichment.csv")
print(f"  - combined_gtopdb_enrichment.csv")
print(f"  - combined_promiscuity_analysis.csv")
print(f"  - combined_promiscuity_plot.png")
print(f"  - combined_top_enriched.png")
print(f"  - combined_summary.txt")