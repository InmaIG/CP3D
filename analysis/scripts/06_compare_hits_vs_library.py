"""
06_compare_hits_vs_library.py
=============================
Compara los 15 hits 3D (12 confirmed + 3 highly toxic) con el resto de
compuestos de C2386 que se testaron pero NO salieron como hits ("lost in 3D").

Como TODA la libreria son bioactivos preseleccionados de un screening 2D,
la pregunta correcta no es "quien es bioactivo" sino "que MoAs/targets/clases
son mas activos en 3D vs los que pierden actividad al pasar a 3D".

Inputs:
- data\\processed\\confirmed_hits_<plate>.csv         (output del 04_)
- C:\\Users\\Ianezi\\Documents\\CP3D\\EOS_compounds_MoA.csv  (library annotations)

Outputs:
- results\\enrichment\\<plate>_C2386_compounds_annotated.csv   (todos los compuestos con hit_status + MoA)
- results\\enrichment\\<plate>_moa_enrichment.csv              (test Fisher por MoA)
- results\\enrichment\\<plate>_target_enrichment.csv           (test Fisher por target/gene)
- results\\enrichment\\<plate>_chembl_enrichment.csv           (jerarquia ChEMBL)
- results\\enrichment\\<plate>_physicochem_comparison.png       (boxplots MW, LogP, n_targets)
- results\\enrichment\\<plate>_top_enriched.png                 (barplots top enriched MoAs/genes)
- results\\enrichment\\<plate>_disgregating_vs_compact.csv     (comparacion 2 disgregantes vs 9 compactos)
- results\\enrichment\\<plate>_summary.txt

Uso:
    python 06_compare_hits_vs_library.py --plate C2386
"""

import argparse
import sys
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ---------------- Paths ----------------
BASE = Path(r'C:\Users\Ianezi\Documents\CP3D\analysis')
PROC_DIR = BASE / 'data' / 'processed'
OUT_DIR = BASE / 'results' / 'enrichment'
OUT_DIR.mkdir(parents=True, exist_ok=True)

LIBRARY_DEFAULT = Path(r'C:\Users\Ianezi\Documents\CP3D\EOS_compounds_MoA.csv')

# ---------------- Args ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--plate', required=True)
parser.add_argument('--library_path', default=str(LIBRARY_DEFAULT))
parser.add_argument('--min_count', type=int, default=2,
                    help='Min total occurrences to include in enrichment test')
parser.add_argument('--min_replicates', type=int, default=3,
                    help='Minimum N_replicates to consider a hit (default: 3)')
args = parser.parse_args()

plate = args.plate
library_path = Path(args.library_path)
min_replicates = args.min_replicates

print(f"\n{'='*70}\n=== HITS vs LIBRARY: {plate} ===\n{'='*70}")
print(f"Library: {library_path}")
print(f"Min replicates filter: {min_replicates}")

# ---------------- Load ----------------
hits_path = PROC_DIR / f'confirmed_hits_{plate}.csv'
if not hits_path.exists():
    print(f"\nERROR: No se encuentra {hits_path}")
    sys.exit(1)
if not library_path.exists():
    print(f"\nERROR: No se encuentra {library_path}")
    sys.exit(1)

# Load hits CSV
hits_df = pd.read_csv(hits_path)
print(f"\nHits dataframe: {len(hits_df)} compounds (all of plate)")

# Load library (try utf-8, then latin-1)
try:
    library = pd.read_csv(library_path, encoding='utf-8-sig', low_memory=False)
except UnicodeDecodeError:
    library = pd.read_csv(library_path, encoding='latin-1', low_memory=False)
library.columns = [c.replace('\ufeff', '').strip() for c in library.columns]
print(f"Library: {len(library)} compounds")

# ---------------- Filter library to current plate ----------------
plate_lib = library[library['Plate'] == plate].copy()
print(f"Compounds in plate {plate} (in library): {len(plate_lib)}")

# ---------------- Categorize compounds ----------------
print(f"\n{'-'*70}\nCategorize all compounds in plate (with min_replicates filter)\n{'-'*70}")

# Hits = confirmed + highly toxic, BUT ONLY if N_replicates >= min_replicates
# Hits with N < min_replicates are downgraded to 'excluded_low_reps'
def categorize(row):
    is_confirmed = row.get('is_confirmed_hit', False)
    is_active = row.get('is_active_top5pct', False)
    n_reps = row.get('Metadata_n_replicates', 0)
    rep_corr = row.get('Metadata_Replicate_corr', np.nan)
    
    # First identify candidate hits
    is_candidate = False
    candidate_type = ''
    if is_confirmed:
        is_candidate = True
        candidate_type = 'confirmed_hit'
    elif is_active and pd.isna(rep_corr) and n_reps == 1:
        is_candidate = True
        candidate_type = 'highly_toxic_no_replicates'
    
    if not is_candidate:
        return 'lost_in_3D'
    
    # Apply min_replicates filter to candidate hits
    if n_reps < min_replicates:
        return 'excluded_low_reps'  # Downgraded — does NOT count as hit
    
    return candidate_type

hits_df['Hit_status'] = hits_df.apply(categorize, axis=1)

# Report excluded hits before continuing
n_excluded = (hits_df['Hit_status'] == 'excluded_low_reps').sum()
if n_excluded > 0:
    print(f"  {n_excluded} candidate hits excluded due to N_replicates < {min_replicates}:")
    excl = hits_df[hits_df['Hit_status']=='excluded_low_reps']
    for _, r in excl.iterrows():
        print(f"    {r['Metadata_Compound']}: N={int(r['Metadata_n_replicates'])}, Activity={r['Metadata_Activity']:.2f}")

# Make a status map: EOS_id -> Hit_status
hit_status_map = dict(zip(hits_df['Metadata_Compound'], hits_df['Hit_status']))

# Add to library subset
plate_lib['Hit_status'] = plate_lib['EOS'].map(hit_status_map).fillna('lost_in_3D')
# Note: excluded_low_reps will NOT be considered as hits in enrichment
plate_lib['Is_3D_hit'] = plate_lib['Hit_status'].isin(['confirmed_hit', 'highly_toxic_no_replicates'])

# Add the metrics from hits_df
plate_lib = plate_lib.merge(
    hits_df[['Metadata_Compound','Metadata_Activity','Metadata_Replicate_corr',
             'Metadata_n_replicates','AreaShape_Area_consensus']].rename(
        columns={'Metadata_Compound':'EOS',
                 'Metadata_Activity':'Activity_score',
                 'Metadata_Replicate_corr':'Replicate_correlation',
                 'Metadata_n_replicates':'N_replicates',
                 'AreaShape_Area_consensus':'AreaShape_Area_zscore'}),
    on='EOS', how='left')

# Phenotype for 3D hits
def assign_phenotype(row):
    if not row['Is_3D_hit']:
        return ''
    area = row.get('AreaShape_Area_zscore', np.nan)
    if pd.isna(area):
        return 'unknown'
    if area > 1:
        return 'expanded'
    elif area < -1:
        return 'shrunken'
    else:
        return 'stable'

plate_lib['Phenotype'] = plate_lib.apply(assign_phenotype, axis=1)

n_hits = plate_lib['Is_3D_hit'].sum()
n_excluded_in_lib = (plate_lib['Hit_status'] == 'excluded_low_reps').sum()
n_lost = len(plate_lib) - n_hits - n_excluded_in_lib
print(f"  3D hits (downstream): {n_hits}")
print(f"    confirmed_hit:                   {(plate_lib['Hit_status']=='confirmed_hit').sum()}")
print(f"    highly_toxic_no_replicates:     {(plate_lib['Hit_status']=='highly_toxic_no_replicates').sum()}")
if n_excluded_in_lib > 0:
    print(f"  Excluded (low replicates, not used): {n_excluded_in_lib}")
print(f"  Lost in 3D: {n_lost}")

# Save annotated table
out_annot = OUT_DIR / f'{plate}_compounds_annotated.csv'
key_cols = ['EOS','EUopen_name','Hit_status','Is_3D_hit','Phenotype',
            'Activity_score','Replicate_correlation','N_replicates',
            'AreaShape_Area_zscore',
            'EUopen_no. targets','EUopen_target_name','EUopen_gene_name',
            'EUopen_target_type','EUopen_moa','EUopen_mw',
            'EUopen_GTOPDB [LEVEL 1]','EUopen_GTOPDB [LEVEL 2]',
            'EUopen_ChEMBL [LEVEL 1]','EUopen_ChEMBL [LEVEL 2]',
            'EUopen_Reactome [LEVEL 1]','Bioactive_Plate','Plate','Well']
key_cols = [c for c in key_cols if c in plate_lib.columns]
plate_lib[key_cols].to_csv(out_annot, index=False)
print(f"\n  Annotated table saved: {out_annot}")

# ---------------- Helpers for enrichment ----------------

def split_multi(text, sep=';'):
    """Split a semicolon-separated string into items, trim whitespace, drop empties."""
    if pd.isna(text):
        return []
    items = [x.strip() for x in str(text).split(sep)]
    return [x for x in items if x]


def enrichment_test(items_hits, items_lost, n_hits_total, n_lost_total, min_count=2):
    """For each unique item, do a Fisher exact test for hits vs lost."""
    # Count occurrences (a compound contributes 1 per unique item)
    hits_counter = Counter()
    for items in items_hits:
        for it in set(items):
            hits_counter[it] += 1
    
    lost_counter = Counter()
    for items in items_lost:
        for it in set(items):
            lost_counter[it] += 1
    
    all_items = set(hits_counter) | set(lost_counter)
    rows = []
    for item in all_items:
        a = hits_counter.get(item, 0)        # in hits with this item
        b = n_hits_total - a                  # in hits without this item
        c = lost_counter.get(item, 0)        # in lost with this item
        d = n_lost_total - c                  # in lost without this item
        
        total = a + c
        if total < min_count:
            continue
        
        # Fisher exact (greater = enriched in hits)
        try:
            odds_ratio, p_greater = stats.fisher_exact(
                [[a, b], [c, d]], alternative='greater')
            _, p_less = stats.fisher_exact(
                [[a, b], [c, d]], alternative='less')
            _, p_two = stats.fisher_exact(
                [[a, b], [c, d]], alternative='two-sided')
        except Exception:
            continue
        
        rows.append({
            'item': item,
            'hits_count': a,
            'hits_pct': 100*a/n_hits_total if n_hits_total > 0 else 0,
            'lost_count': c,
            'lost_pct': 100*c/n_lost_total if n_lost_total > 0 else 0,
            'odds_ratio': odds_ratio,
            'p_value': p_greater,  # one-tailed: enriched in hits
            'p_value_two_sided': p_two,
        })
    
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values('p_value')
    
    # Benjamini-Hochberg FDR
    if len(df) > 0:
        from scipy.stats import false_discovery_control
        try:
            df['fdr'] = false_discovery_control(df['p_value'].values)
        except Exception:
            # Fallback manual BH
            n = len(df)
            ranks = np.arange(1, n+1)
            df_sorted = df.sort_values('p_value').reset_index(drop=True)
            df_sorted['fdr'] = (df_sorted['p_value'] * n / ranks).clip(0, 1)
            # Make monotone
            df_sorted['fdr'] = df_sorted['fdr'].iloc[::-1].cummin().iloc[::-1]
            df = df_sorted
    return df


# ---------------- Enrichment: MoA ----------------
print(f"\n{'-'*70}\nEnrichment analysis: MoA\n{'-'*70}")

hits_subset = plate_lib[plate_lib['Is_3D_hit']]
# Lost in 3D = compounds tested but NOT hits AND NOT excluded by low replicates
# (excluded_low_reps are removed from analysis entirely — they're hits we couldn't validate)
lost_subset = plate_lib[(~plate_lib['Is_3D_hit']) & (plate_lib['Hit_status'] != 'excluded_low_reps')]

moa_hits = hits_subset['EUopen_moa'].apply(split_multi).tolist()
moa_lost = lost_subset['EUopen_moa'].apply(split_multi).tolist()

moa_enrich = enrichment_test(moa_hits, moa_lost, len(hits_subset), len(lost_subset),
                              min_count=args.min_count)
if len(moa_enrich) > 0:
    print(f"  Tested {len(moa_enrich)} MoAs")
    print(f"  Top 10 by p-value (enriched in hits):")
    print(moa_enrich.head(10).to_string(index=False))
    out = OUT_DIR / f'{plate}_moa_enrichment.csv'
    moa_enrich.to_csv(out, index=False)
    print(f"\n  Saved: {out}")
else:
    print("  No MoAs to test")

# ---------------- Enrichment: target_name (split per protein) ----------------
print(f"\n{'-'*70}\nEnrichment analysis: target_name\n{'-'*70}")

target_hits = hits_subset['EUopen_target_name'].apply(split_multi).tolist()
target_lost = lost_subset['EUopen_target_name'].apply(split_multi).tolist()

target_enrich = enrichment_test(target_hits, target_lost,
                                 len(hits_subset), len(lost_subset),
                                 min_count=args.min_count)
if len(target_enrich) > 0:
    print(f"  Tested {len(target_enrich)} unique targets")
    print(f"  Top 10 by p-value:")
    print(target_enrich.head(10).to_string(index=False))
    out = OUT_DIR / f'{plate}_target_enrichment.csv'
    target_enrich.to_csv(out, index=False)
    print(f"\n  Saved: {out}")

# ---------------- Enrichment: gene_name ----------------
print(f"\n{'-'*70}\nEnrichment analysis: gene_name\n{'-'*70}")

gene_hits = hits_subset['EUopen_gene_name'].apply(split_multi).tolist()
gene_lost = lost_subset['EUopen_gene_name'].apply(split_multi).tolist()

gene_enrich = enrichment_test(gene_hits, gene_lost,
                              len(hits_subset), len(lost_subset),
                              min_count=args.min_count)
if len(gene_enrich) > 0:
    print(f"  Tested {len(gene_enrich)} unique genes")
    print(f"  Top 10 enriched genes:")
    print(gene_enrich.head(10).to_string(index=False))
    out = OUT_DIR / f'{plate}_gene_enrichment.csv'
    gene_enrich.to_csv(out, index=False)

# ---------------- Enrichment: ChEMBL hierarchy ----------------
print(f"\n{'-'*70}\nEnrichment analysis: ChEMBL classification levels\n{'-'*70}")

chembl_results = {}
for level in [1, 2, 3, 4]:
    col = f'EUopen_ChEMBL [LEVEL {level}]'
    if col not in plate_lib.columns:
        continue
    h = hits_subset[col].apply(split_multi).tolist()
    l = lost_subset[col].apply(split_multi).tolist()
    res = enrichment_test(h, l, len(hits_subset), len(lost_subset),
                          min_count=args.min_count)
    if len(res) > 0:
        res['chembl_level'] = level
        chembl_results[level] = res
        print(f"\n  Level {level}: {len(res)} tested. Top 5:")
        print(res.head(5).to_string(index=False))

if chembl_results:
    chembl_combined = pd.concat(chembl_results.values(), ignore_index=True)
    chembl_combined = chembl_combined.sort_values('p_value')
    out = OUT_DIR / f'{plate}_chembl_enrichment.csv'
    chembl_combined.to_csv(out, index=False)
    print(f"\n  Saved combined ChEMBL enrichment: {out}")

# ---------------- Enrichment: GTOPDB classification ----------------
print(f"\n{'-'*70}\nEnrichment analysis: GTOPDB classification\n{'-'*70}")

gtopdb_results = {}
for level in [1, 2, 3]:
    col = f'EUopen_GTOPDB [LEVEL {level}]'
    if col not in plate_lib.columns:
        continue
    h = hits_subset[col].apply(split_multi).tolist()
    l = lost_subset[col].apply(split_multi).tolist()
    res = enrichment_test(h, l, len(hits_subset), len(lost_subset),
                          min_count=args.min_count)
    if len(res) > 0:
        res['gtopdb_level'] = level
        gtopdb_results[level] = res
        print(f"\n  Level {level}: top 5:")
        print(res.head(5).to_string(index=False))

if gtopdb_results:
    gtopdb_combined = pd.concat(gtopdb_results.values(), ignore_index=True)
    gtopdb_combined = gtopdb_combined.sort_values('p_value')
    out = OUT_DIR / f'{plate}_gtopdb_enrichment.csv'
    gtopdb_combined.to_csv(out, index=False)

# ---------------- Physicochemical comparison ----------------
print(f"\n{'-'*70}\nPhysicochemical comparison\n{'-'*70}")

phys_cols = {
    'EUopen_mw': 'Molecular weight',
    'EUopen_no. targets': 'Number of targets',
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, (col, label) in enumerate(phys_cols.items()):
    if col not in plate_lib.columns:
        continue
    ax = axes[i]
    h_vals = pd.to_numeric(hits_subset[col], errors='coerce').dropna()
    l_vals = pd.to_numeric(lost_subset[col], errors='coerce').dropna()
    
    if len(h_vals) > 0 and len(l_vals) > 0:
        # Mann-Whitney U test (non-parametric)
        u_stat, u_p = stats.mannwhitneyu(h_vals, l_vals, alternative='two-sided')
        bp = ax.boxplot([h_vals, l_vals], labels=['3D hits', 'Lost in 3D'],
                        patch_artist=True, showfliers=True)
        bp['boxes'][0].set_facecolor('#E74C3C')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('#95A5A6')
        bp['boxes'][1].set_alpha(0.7)
        ax.set_ylabel(label)
        ax.set_title(f'{label}\nMann-Whitney U p={u_p:.3g}')
        ax.grid(alpha=0.3, axis='y')
        
        # Print stats
        print(f"\n  {label}:")
        print(f"    3D hits (n={len(h_vals)}): median={h_vals.median():.2f}, "
              f"mean={h_vals.mean():.2f}")
        print(f"    Lost in 3D (n={len(l_vals)}): median={l_vals.median():.2f}, "
              f"mean={l_vals.mean():.2f}")
        print(f"    Mann-Whitney U: p={u_p:.4f}")

plt.suptitle(f'Physicochemical comparison: {plate}', fontweight='bold')
plt.tight_layout()
out_phys = OUT_DIR / f'{plate}_physicochem_comparison.png'
plt.savefig(out_phys, dpi=120, bbox_inches='tight')
plt.close()
print(f"\n  Plot saved: {out_phys}")

# ---------------- Top enriched plot ----------------
print(f"\n{'-'*70}\nTop enriched plot\n{'-'*70}")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# MoA enrichment
ax = axes[0]
if len(moa_enrich) > 0:
    top_moa = moa_enrich.head(10)
    y_pos = np.arange(len(top_moa))
    bars = ax.barh(y_pos, top_moa['hits_pct'], color='#E74C3C', alpha=0.7,
                   label='3D hits %')
    ax.barh(y_pos, top_moa['lost_pct'], color='#95A5A6', alpha=0.5,
            label='Lost in 3D %')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_moa['item'].apply(lambda x: x[:30]))
    ax.invert_yaxis()
    ax.set_xlabel('% of compounds')
    ax.set_title(f'Top 10 MoAs\n(by p-value, enriched in hits)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3, axis='x')

# Gene enrichment
ax = axes[1]
if len(gene_enrich) > 0:
    top_gene = gene_enrich.head(15)
    y_pos = np.arange(len(top_gene))
    ax.barh(y_pos, top_gene['hits_pct'], color='#E74C3C', alpha=0.7, label='3D hits %')
    ax.barh(y_pos, top_gene['lost_pct'], color='#95A5A6', alpha=0.5, label='Lost in 3D %')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_gene['item'].apply(lambda x: x[:25]))
    ax.invert_yaxis()
    ax.set_xlabel('% of compounds')
    ax.set_title(f'Top 15 genes/targets\n(by p-value, enriched in hits)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3, axis='x')

# ChEMBL Level 2 (functional class)
ax = axes[2]
if 2 in chembl_results and len(chembl_results[2]) > 0:
    top_chembl = chembl_results[2].head(10)
    y_pos = np.arange(len(top_chembl))
    ax.barh(y_pos, top_chembl['hits_pct'], color='#E74C3C', alpha=0.7, label='3D hits %')
    ax.barh(y_pos, top_chembl['lost_pct'], color='#95A5A6', alpha=0.5, label='Lost in 3D %')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_chembl['item'].apply(lambda x: x[:30]))
    ax.invert_yaxis()
    ax.set_xlabel('% of compounds')
    ax.set_title(f'Top 10 ChEMBL Level 2\n(functional class)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3, axis='x')

plt.suptitle(f'Top enriched in 3D hits: {plate}', fontweight='bold')
plt.tight_layout()
out_top = OUT_DIR / f'{plate}_top_enriched.png'
plt.savefig(out_top, dpi=120, bbox_inches='tight')
plt.close()
print(f"  Plot saved: {out_top}")

# ---------------- Expanded vs shrunken ----------------
print(f"\n{'-'*70}\nExpanded vs shrunken comparison\n{'-'*70}")

expanded_df = plate_lib[plate_lib['Phenotype']=='expanded']
shrunken_df = plate_lib[plate_lib['Phenotype']=='shrunken']
print(f"  Expanded: {len(expanded_df)} hits")
print(f"  Cytotoxic compact: {len(shrunken_df)} hits")

if len(expanded_df) > 0 and len(shrunken_df) > 0:
    cmp_cols = ['EOS','EUopen_name','Activity_score','Replicate_correlation',
                'AreaShape_Area_zscore','Phenotype',
                'EUopen_no. targets','EUopen_target_name','EUopen_gene_name',
                'EUopen_moa','EUopen_ChEMBL [LEVEL 1]','EUopen_ChEMBL [LEVEL 2]']
    cmp_cols = [c for c in cmp_cols if c in plate_lib.columns]
    cmp_df = pd.concat([expanded_df, shrunken_df])[cmp_cols].copy()
    cmp_df = cmp_df.sort_values(['Phenotype','Activity_score'], ascending=[False, False])
    out_cmp = OUT_DIR / f'{plate}_disgregating_vs_compact.csv'
    cmp_df.to_csv(out_cmp, index=False)
    print(f"  Saved: {out_cmp}")

# ---------------- Summary ----------------
summary = OUT_DIR / f'{plate}_summary.txt'
with open(summary, 'w') as f:
    f.write(f"Hits vs Library comparison: {plate}\n")
    f.write(f"="*70 + "\n\n")
    f.write(f"Compounds in plate: {len(plate_lib)}\n")
    f.write(f"  3D hits (12 confirmed + 3 highly toxic): {n_hits}\n")
    f.write(f"  Lost in 3D: {n_lost}\n\n")
    
    f.write(f"Phenotypes (3D hits only):\n")
    for ph, n in plate_lib[plate_lib['Is_3D_hit']]['Phenotype'].value_counts().items():
        f.write(f"  {ph}: {n}\n")
    
    f.write(f"\n--- Top enriched MoAs (Fisher exact, FDR < 0.25) ---\n")
    if len(moa_enrich) > 0:
        sig = moa_enrich[moa_enrich['fdr'] < 0.25]
        if len(sig) > 0:
            for _, r in sig.head(15).iterrows():
                f.write(f"  {r['item']}: hits={r['hits_count']}/{n_hits} ({r['hits_pct']:.0f}%), "
                        f"lost={r['lost_count']}/{n_lost} ({r['lost_pct']:.1f}%), "
                        f"OR={r['odds_ratio']:.2f}, p={r['p_value']:.3g}, fdr={r['fdr']:.3g}\n")
        else:
            f.write("  No MoAs with FDR < 0.25\n")
            f.write(f"  Top 5 by p-value:\n")
            for _, r in moa_enrich.head(5).iterrows():
                f.write(f"  {r['item']}: hits={r['hits_count']}/{n_hits}, "
                        f"lost={r['lost_count']}/{n_lost}, p={r['p_value']:.3g}\n")
    
    f.write(f"\n--- Top enriched genes/targets ---\n")
    if len(gene_enrich) > 0:
        for _, r in gene_enrich.head(15).iterrows():
            f.write(f"  {r['item']}: hits={r['hits_count']}/{n_hits}, "
                    f"lost={r['lost_count']}/{n_lost}, p={r['p_value']:.3g}, fdr={r['fdr']:.3g}\n")
    
    f.write(f"\n--- Physicochemical (Mann-Whitney) ---\n")
    for col, label in phys_cols.items():
        if col in plate_lib.columns:
            h = pd.to_numeric(hits_subset[col], errors='coerce').dropna()
            l = pd.to_numeric(lost_subset[col], errors='coerce').dropna()
            if len(h) > 0 and len(l) > 0:
                _, p = stats.mannwhitneyu(h, l, alternative='two-sided')
                f.write(f"  {label}: hits median={h.median():.1f}, "
                        f"lost median={l.median():.1f}, p={p:.4f}\n")

print(f"\n  Summary saved: {summary}")

print(f"\n{'='*70}\n=== DONE ===\n{'='*70}")
print(f"\nKey outputs in {OUT_DIR}:")
print(f"  - {plate}_compounds_annotated.csv     (all compounds with hit/MoA info)")
print(f"  - {plate}_moa_enrichment.csv          (Fisher test by MoA)")
print(f"  - {plate}_target_enrichment.csv       (Fisher test by target protein)")
print(f"  - {plate}_gene_enrichment.csv         (Fisher test by gene)")
print(f"  - {plate}_chembl_enrichment.csv       (ChEMBL functional class)")
print(f"  - {plate}_gtopdb_enrichment.csv       (GTOPDB pharmacological class)")
print(f"  - {plate}_top_enriched.png            (visual summary)")
print(f"  - {plate}_physicochem_comparison.png  (MW, n_targets boxplots)")
print(f"  - {plate}_disgregating_vs_compact.csv (special comparison)")
print(f"  - {plate}_summary.txt                 (text summary)")