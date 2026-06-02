"""
21_lincs_l1000_validation.py  (FINAL VERSION using SigCom LINCS / LDP3)
========================================================================
Transcriptomic orthogonal validation of the HSP90 chemotype-specific phenotype
using LINCS L1000 signatures from the SigCom LINCS API (LINCS Data Portal 3,
Mt. Sinai / MaayanLab), accessed via the LINCS-DCIC S3 bucket which serves
the underlying TSV signature files directly.

Pipeline
--------
1. For each HSP family compound: query SigCom /signatures/find by pert_name
   (lowercase exact match).
2. Filter signatures to: cell_line in [HEPG2, PHH, HUH7] (liver-relevant)
   with fallback to [A549, MCF7] (most-covered cell lines).
3. Preferred dose 10 uM, preferred time 24 h.
4. Download the per-gene Characteristic Direction (CD) coefficient TSV via
   meta.persistent_id (S3 public).
5. Convert CD-coefficients to within-signature percentile ranks (the
   biologically meaningful quantity for CD values, which are unit-vector
   loadings bounded ~[-0.06, +0.06]).
6. Extract percentile ranks for canonical HSF1 target genes (heat shock
   response) and compute composite HSR score per compound x cell line.
7. Compare chemotypes (ansamycins vs others).
8. Correlate transcriptomic HSR with phenotypic 3D activity.

Heat shock response gene set (HSF1 targets, canonical)
------------------------------------------------------
HSPA1A, HSPA1B, HSPA6, HSPA8     HSP70 family (induced by HSF1)
HSPH1, HSPB1                       HSP110, HSP27
DNAJB1, DNAJB4, DNAJB9             HSP40 co-chaperones
BAG3                                HSF1 target classical
SERPINH1                            HSF1 target inducible
HSPE1, HSPD1                       Mitochondrial chaperones
DNAJC3                              ER stress / DNAJ

Outputs (results/lincs_l1000/)
------------------------------
compound_signature_metadata.csv    Available signatures per compound
heat_shock_scores.csv              HSR composite score per compound
heat_shock_signatures_raw.csv      Per-gene CD percentile ranks per compound
figures/
    hsr_by_compound.{png,pdf,svg}
    hsr_heatmap.{png,pdf,svg}
    hsr_vs_3d_activity.{png,pdf,svg}
summary.txt

Interpretation of HSR score
---------------------------
HSR_mean_pctrank ~ 50  -> heat shock genes at neutral / average rank
HSR_mean_pctrank > 75  -> heat shock genes enriched in top quartile (induction)
HSR_mean_pctrank > 90  -> strong heat shock response (HSF1 activation)
HSR_mean_pctrank < 25  -> heat shock genes repressed below baseline

Usage
-----
    python 21_lincs_l1000_validation.py
"""

import io
import sys
import time
from pathlib import Path
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(r'C:\Users\Ianezi\Documents\CP3D')
ANALYSIS = BASE / 'analysis'
OUT = ANALYSIS / 'results' / 'lincs_l1000'
FIG = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
PER_COMPOUND = ANALYSIS / 'results' / 'medina_2d_vs_3d' / 'per_compound.csv'

SIGCOM_META = 'https://maayanlab.cloud/sigcom-lincs/metadata-api'

# Compounds of interest with possible name aliases used by LINCS
COMPOUND_QUERIES = {
    'GELDANAMYCIN':  {'aliases': ['geldanamycin'],
                       'chemotype': 'Ansamycin (HSP90)',
                       'redox': 'high', 'is_hit_3D': True},
    'ALVESPIMYCIN':  {'aliases': ['alvespimycin', '17-dmag'],
                       'chemotype': 'Ansamycin (HSP90)',
                       'redox': 'high', 'is_hit_3D': True},
    'RETASPIMYCIN':  {'aliases': ['retaspimycin', 'ipi-504'],
                       'chemotype': 'Ansamycin (HSP90)',
                       'redox': 'medium-high', 'is_hit_3D': True},
    'TANESPIMYCIN':  {'aliases': ['tanespimycin', '17-aag',
                                    '17-(allylamino)-17-demethoxygeldanamycin'],
                       'chemotype': 'Ansamycin (HSP90)',
                       'redox': 'attenuated', 'is_hit_3D': False},
    'LUMINESPIB':    {'aliases': ['luminespib', 'auy922', 'nvp-auy922'],
                       'chemotype': 'Resorcinol (HSP90)',
                       'redox': 'low', 'is_hit_3D': False},
    'GANETESPIB':    {'aliases': ['ganetespib', 'sta-9090'],
                       'chemotype': 'Resorcinol (HSP90)',
                       'redox': 'low', 'is_hit_3D': False},
    'BIIB021':       {'aliases': ['biib021', 'biib-021'],
                       'chemotype': 'Purine (HSP90)',
                       'redox': 'low', 'is_hit_3D': False},
    'SNX-2112':      {'aliases': ['snx-2112', 'snx2112'],
                       'chemotype': 'Benzamide (HSP90)',
                       'redox': 'low', 'is_hit_3D': False},
    'VER-155008':    {'aliases': ['ver-155008', 'ver155008'],
                       'chemotype': 'HSP70 inhibitor',
                       'redox': 'low', 'is_hit_3D': False},
    'RADICICOL':     {'aliases': ['radicicol'],
                       'chemotype': 'Resorcylic lactam (HSP90 positive control)',
                       'redox': 'low', 'is_hit_3D': False},
}

CHEMOTYPE_PALETTE = {
    'Ansamycin (HSP90)':                                       '#D7263D',
    'Resorcinol (HSP90)':                                       '#1F77B4',
    'Purine (HSP90)':                                           '#2CA02C',
    'Benzamide (HSP90)':                                        '#9467BD',
    'HSP70 inhibitor':                                          '#7F7F7F',
    'Resorcylic lactam (HSP90 positive control)':               '#FF7F0E',
}

HEAT_SHOCK_GENES = [
    'HSPA1A', 'HSPA1B', 'HSPA6', 'HSPA8',     # HSP70 family
    'HSPH1', 'HSPB1',                          # HSP110, HSP27
    'DNAJB1', 'DNAJB4', 'DNAJB9',              # HSP40 co-chaperones
    'BAG3', 'SERPINH1',                        # HSF1 target classical
    'HSPE1', 'HSPD1',                          # Mitochondrial chaperones
    'DNAJC3',                                   # ER stress / DNAJ
]

PREFERRED_CELL_LINES = ['HEPG2', 'PHH', 'HUH7', 'A549', 'MCF7', 'PC3', 'VCAP']
PREFERRED_DOSE_PATTERNS = ['10 uM', '10uM', '10 µM']
PREFERRED_TIME_PATTERNS = ['24 h', '24h']

TIMEOUT = 30

def banner(s):
    print(f"\n{'='*70}\n{s}\n{'='*70}")

# ---------------------------------------------------------------------------
# 1. For each compound, find signatures
# ---------------------------------------------------------------------------
banner('1. Querying SigCom /signatures/find for each HSP family compound')

all_meta = []
for compound, info in COMPOUND_QUERIES.items():
    print(f"\n  {compound}")
    found = []
    for alias in info['aliases']:
        try:
            r = requests.post(f'{SIGCOM_META}/signatures/find',
                              json={'filter': {'where': {'meta.pert_name': alias},
                                                'limit': 200}},
                              timeout=TIMEOUT)
            if r.ok:
                sigs = r.json()
                if isinstance(sigs, list) and len(sigs) > 0:
                    print(f"    alias '{alias}': {len(sigs)} signatures")
                    found.extend(sigs)
                else:
                    print(f"    alias '{alias}': 0 signatures")
            else:
                print(f"    alias '{alias}': HTTP {r.status_code}")
        except Exception as e:
            print(f"    alias '{alias}': ERROR {e}")
        time.sleep(0.3)
    if not found:
        print(f"    -- {compound} NOT found in SigCom with any alias")
        continue
    # Deduplicate by signature id
    unique = {}
    for s in found:
        unique[s.get('id')] = s
    found_unique = list(unique.values())
    print(f"    Total unique signatures: {len(found_unique)}")
    # Add compound info
    for s in found_unique:
        m = s.get('meta', {})
        all_meta.append({
            'compound': compound,
            'chemotype': info['chemotype'],
            'redox': info['redox'],
            'is_hit_3D': info['is_hit_3D'],
            'signature_id': s.get('id'),
            'cell_line': m.get('cell_line', '').upper(),
            'tissue': m.get('tissue'),
            'pert_dose': m.get('pert_dose'),
            'pert_time': m.get('pert_time'),
            'pert_name': m.get('pert_name'),
            'persistent_id': m.get('persistent_id'),
        })

meta_df = pd.DataFrame(all_meta)
meta_df.to_csv(OUT / 'compound_signature_metadata.csv', index=False)
print(f"\n  Total signatures collected: {len(meta_df)}")
print(f"\n  Cell line coverage per compound:")
if not meta_df.empty:
    coverage = meta_df.groupby(['compound', 'cell_line']).size().unstack(fill_value=0)
    print(coverage.to_string())

if meta_df.empty:
    print("\n  ERROR: no signatures recovered. Exiting.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Select one preferred signature per compound
# ---------------------------------------------------------------------------
banner('2. Selecting one preferred signature per compound')

def select_best(sub):
    """Choose best signature: HEPG2 > liver-relevant > A549 > MCF7,
    then prefer 10 uM and 24 h."""
    if sub.empty:
        return None
    for cl in PREFERRED_CELL_LINES:
        cl_sub = sub[sub['cell_line'] == cl]
        if cl_sub.empty:
            continue
        # Prefer dose 10 uM
        for dpat in PREFERRED_DOSE_PATTERNS:
            d = cl_sub[cl_sub['pert_dose'].fillna('').str.contains(dpat,
                                                                     case=False, regex=False)]
            if not d.empty:
                for tpat in PREFERRED_TIME_PATTERNS:
                    t = d[d['pert_time'].fillna('').str.contains(tpat,
                                                                   case=False, regex=False)]
                    if not t.empty:
                        return t.iloc[0]
                return d.iloc[0]
        return cl_sub.iloc[0]
    return sub.iloc[0]

selected = []
for compound in COMPOUND_QUERIES:
    sub = meta_df[meta_df['compound'] == compound]
    chosen = select_best(sub)
    if chosen is None:
        print(f"  {compound:14s}: NOT AVAILABLE")
        continue
    selected.append(chosen)
    print(f"  {compound:14s}: cell={chosen['cell_line']:10s}, "
          f"dose={chosen['pert_dose']}, time={chosen['pert_time']}, "
          f"id={str(chosen['signature_id'])[:36]}")

sel_df = pd.DataFrame(selected)

# ---------------------------------------------------------------------------
# 3. Download the TSV file from persistent_id
# ---------------------------------------------------------------------------
banner('3. Downloading per-gene z-scores from S3 (persistent_id)')

def parse_tsv_signature(tsv_text):
    """Parse a LINCS L1000 signature TSV (CD-coefficient format from SigCom).

    The TSV has 2 columns: gene_symbol and CD-coefficient (Characteristic
    Direction value, Clark et al. 2014). CD coefficients are unit-vector
    loadings, naturally bounded ~[-0.06, +0.06]. The biologically meaningful
    quantity is the RANK of each gene within the signature, not the absolute
    value.

    Returns a DataFrame with columns: gene_symbol, value, percentile_rank.
    `value` keeps the raw CD coefficient; `percentile_rank` is the gene's
    rank within the signature's distribution (0-100, where 100 = most induced,
    0 = most repressed, 50 = neutral)."""
    try:
        df = pd.read_csv(io.StringIO(tsv_text), sep='\t')
    except Exception:
        return None
    gene_col = None
    for c in df.columns:
        cl = str(c).lower()
        if cl in ('gene', 'symbol', 'gene_symbol', 'pr_gene_symbol',
                  'genesymbol'):
            gene_col = c
            break
    value_col = None
    for c in df.columns:
        cl = str(c).lower()
        if cl in ('value', 'zscore', 'z_score', 'z-score', 'l1000_value',
                  'modz', 'z', 'cd-coefficient', 'cd_coefficient',
                  'cd coefficient', 'cd'):
            value_col = c
            break
    if gene_col is None:
        gene_col = df.columns[0]
    if value_col is None and df.shape[1] >= 2:
        value_col = df.columns[1]
    if gene_col is None or value_col is None:
        return None
    out = df[[gene_col, value_col]].copy()
    out.columns = ['gene_symbol', 'value']
    out['gene_symbol'] = out['gene_symbol'].astype(str).str.upper().str.strip()
    out = out.dropna()
    # Compute percentile rank within the signature distribution
    out['percentile_rank'] = out['value'].rank(pct=True) * 100
    return out

per_compound_data = {}
for _, r in sel_df.iterrows():
    compound = r['compound']
    url = r['persistent_id']
    if pd.isna(url) or not str(url).startswith('http'):
        print(f"  {compound:14s}: no persistent_id URL")
        continue
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        if not resp.ok:
            print(f"  {compound:14s}: HTTP {resp.status_code} fetching {url}")
            continue
        sig_df = parse_tsv_signature(resp.text)
        if sig_df is None or len(sig_df) < 100:
            print(f"  {compound:14s}: parsing returned {0 if sig_df is None else len(sig_df)} rows")
            print(f"    First 200 chars of file: {resp.text[:200]}")
            continue
        per_compound_data[compound] = sig_df
        print(f"  {compound:14s}: {len(sig_df)} gene z-scores downloaded")
    except Exception as e:
        print(f"  {compound:14s}: ERROR {e}")
    time.sleep(0.3)

if not per_compound_data:
    print("\n  ERROR: no signature data downloaded. Check S3 access.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 4. Extract heat shock response percentile ranks and compute composite score
# ---------------------------------------------------------------------------
banner('4. Computing Heat Shock Response (HSR) composite scores')
print('  HSR score = mean within-signature percentile rank of HSF1 target')
print('  genes. Range 0-100; 50 = neutral; >75 = enriched at top of signature;')
print('  >90 = strong induction; <25 = repressed.\n')

hs_rows = []
hs_per_gene = {}        # percentile ranks (used downstream)
hs_per_gene_cd = {}     # raw CD coefficients (kept for reference)
for compound, sig_df in per_compound_data.items():
    chosen = sel_df[sel_df['compound'] == compound].iloc[0]
    info = COMPOUND_QUERIES[compound]
    # Get percentile ranks (and raw CD) for heat shock genes
    hs_genes_present = sig_df[sig_df['gene_symbol'].isin(HEAT_SHOCK_GENES)]
    hs_ranks = dict(zip(hs_genes_present['gene_symbol'],
                         hs_genes_present['percentile_rank'].astype(float)))
    hs_cd = dict(zip(hs_genes_present['gene_symbol'],
                      hs_genes_present['value'].astype(float)))
    if not hs_ranks:
        print(f"  {compound:14s}: no heat shock genes found in signature")
        continue
    hs_per_gene[compound] = hs_ranks
    hs_per_gene_cd[compound] = hs_cd
    rank_mean = float(np.mean(list(hs_ranks.values())))
    rank_median = float(np.median(list(hs_ranks.values())))
    cd_mean = float(np.mean(list(hs_cd.values())))
    n_genes = len(hs_ranks)
    hs_rows.append({
        'compound': compound,
        'chemotype': info['chemotype'],
        'redox': info['redox'],
        'is_hit_3D': info['is_hit_3D'],
        'cell_line': chosen['cell_line'],
        'pert_dose': chosen['pert_dose'],
        'pert_time': chosen['pert_time'],
        'signature_id': str(chosen['signature_id']),
        'HSR_mean_pctrank': rank_mean,
        'HSR_median_pctrank': rank_median,
        'HSR_mean_cd_raw': cd_mean,
        'n_heat_shock_genes': n_genes,
    })
    print(f"  {compound:14s}: HSR pct rank = {rank_mean:5.1f} (median {rank_median:5.1f}), "
          f"raw CD mean = {cd_mean:+.4f}, n genes = {n_genes}")

hs_df = pd.DataFrame(hs_rows).sort_values('HSR_mean_pctrank', ascending=False)
hs_df.to_csv(OUT / 'heat_shock_scores.csv', index=False)
print(f"\n  Saved: {OUT / 'heat_shock_scores.csv'}")

# Per-gene matrices (percentile rank used in heatmap; raw CD kept alongside)
if hs_per_gene:
    per_gene_df = pd.DataFrame(hs_per_gene).T
    per_gene_df.to_csv(OUT / 'heat_shock_signatures_raw.csv')
    per_gene_cd_df = pd.DataFrame(hs_per_gene_cd).T
    per_gene_cd_df.to_csv(OUT / 'heat_shock_signatures_cd_coefficients.csv')

# ---------------------------------------------------------------------------
# 5. Compare by chemotype
# ---------------------------------------------------------------------------
banner('5. HSR composite percentile rank by chemotype')
print('  (Reference: 50 = neutral baseline; > 75 = enriched HSR)\n')
for ct in CHEMOTYPE_PALETTE:
    sub = hs_df[hs_df['chemotype'] == ct]
    if not sub.empty:
        n = len(sub)
        m = sub['HSR_mean_pctrank'].mean()
        mx = sub['HSR_mean_pctrank'].max()
        sd = sub['HSR_mean_pctrank'].std() if n > 1 else 0
        print(f"  {ct:50s} n={n}, mean={m:5.1f} +/- {sd:4.1f}, max={mx:5.1f}")

# Ansamycin (hit) vs others one-sided Mann-Whitney
ansamycin_hits = hs_df[(hs_df['chemotype'] == 'Ansamycin (HSP90)') & hs_df['is_hit_3D']]
other_hsps = hs_df[(hs_df['chemotype'] != 'Ansamycin (HSP90)')
                    & hs_df['chemotype'].str.contains('HSP90', na=False)]
if len(ansamycin_hits) >= 2 and len(other_hsps) >= 2:
    try:
        u, p_mw = stats.mannwhitneyu(ansamycin_hits['HSR_mean_pctrank'],
                                       other_hsps['HSR_mean_pctrank'],
                                       alternative='greater')
        print(f"\n  Ansamycin-hits vs non-ansamycin HSP90 inhibitors")
        print(f"    Mann-Whitney U one-sided (ansamycin > others): "
              f"U = {u:.1f}, p = {p_mw:.3g}")
    except Exception as e:
        print(f"  Mann-Whitney failed: {e}")

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
banner('6. Figures')

# Figure 1: HSR by compound bar chart (percentile rank scale 0-100)
fig, ax = plt.subplots(figsize=(11, 7))
plot_df = hs_df.sort_values('HSR_mean_pctrank')
colors = [CHEMOTYPE_PALETTE.get(c, '#888') for c in plot_df['chemotype']]
y = np.arange(len(plot_df))
edges = ['black' if h else 'none' for h in plot_df['is_hit_3D']]
linewidths = [2 if h else 0.5 for h in plot_df['is_hit_3D']]
# Bars are drawn relative to 50 (the neutral baseline)
bar_vals = plot_df['HSR_mean_pctrank'].values - 50
ax.barh(y, bar_vals, left=50, color=colors,
         edgecolor=edges, linewidth=linewidths)
# Compound labels (left side of plot)
xmin, xmax = 0, 100
for i, (_, r) in enumerate(plot_df.iterrows()):
    drug = r['compound']
    weight = 'bold' if r['is_hit_3D'] else 'normal'
    ax.text(-2, i, drug, va='center', ha='right',
             fontsize=10.5, fontweight=weight)
    # Value at bar end
    v = r['HSR_mean_pctrank']
    if v >= 50:
        ax.text(v + 0.8, i, f'{v:.1f}', va='center', ha='left', fontsize=10,
                 fontweight='bold')
    else:
        ax.text(v - 0.8, i, f'{v:.1f}', va='center', ha='right', fontsize=10,
                 fontweight='bold')
ax.set_yticks([])
ax.set_xlim(-25, 110)
ax.axvline(50, color='black', lw=0.8, ls='--', alpha=0.6,
            label='Neutral (rank 50)')
ax.axvline(75, color='#888', lw=0.6, ls=':', alpha=0.5)
ax.text(75.5, -0.7, 'enriched', fontsize=8, color='#666', ha='left')
ax.set_xlabel('Heat Shock Response score\n(mean within-signature percentile rank of HSF1 target genes; 50 = neutral)',
              fontsize=11)
ax.set_title('LINCS L1000 transcriptomic heat shock response across HSP family chemotypes\n'
              'Bold = 3D Cell Painting hits.  Ranks computed within each compound\'s own signature distribution.',
              fontsize=11.5, fontweight='bold', loc='left', pad=10)
legend_patches = [Patch(facecolor=col, edgecolor='black', label=ct)
                   for ct, col in CHEMOTYPE_PALETTE.items()
                   if ct in plot_df['chemotype'].unique()]
ax.legend(handles=legend_patches, loc='lower right', fontsize=9, frameon=True)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
for ext in ('png', 'pdf', 'svg'):
    plt.savefig(FIG / f'hsr_by_compound.{ext}', dpi=200, bbox_inches='tight')
plt.close()
print(f"  Saved hsr_by_compound")

# Figure 2: heatmap (percentile ranks, centered on 50 = neutral)
if len(hs_per_gene) >= 2:
    fig, ax = plt.subplots(figsize=(11, max(5, 0.5 * len(hs_per_gene))))
    ordered_compounds = plot_df['compound'].tolist()
    rows = []
    valid_compounds = []
    for c in ordered_compounds:
        if c in hs_per_gene:
            rows.append([hs_per_gene[c].get(g, np.nan) for g in HEAT_SHOCK_GENES])
            valid_compounds.append(c)
    M = np.array(rows)
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad('#DDDDDD')
    M_masked = np.ma.masked_invalid(M)
    # Diverging scale 0-100 centred on 50
    im = ax.imshow(M_masked, cmap=cmap, vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(HEAT_SHOCK_GENES)))
    ax.set_xticklabels(HEAT_SHOCK_GENES, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(valid_compounds)))
    ax.set_yticklabels(valid_compounds, fontsize=10)
    for i, c in enumerate(valid_compounds):
        ct = COMPOUND_QUERIES[c]['chemotype']
        ax.get_yticklabels()[i].set_color(CHEMOTYPE_PALETTE.get(ct, '#888'))
        if COMPOUND_QUERIES[c]['is_hit_3D']:
            ax.get_yticklabels()[i].set_fontweight('bold')
    # Annotate values
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isnan(v):
                col = 'white' if (v > 75 or v < 25) else 'black'
                ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                         fontsize=8.5, color=col)
            else:
                ax.text(j, i, 'NA', ha='center', va='center', fontsize=7,
                         color='#999')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8,
                         label='Within-signature percentile rank\n(0 = most repressed, 50 = neutral, 100 = most induced)')
    cbar.ax.axhline(50, color='black', lw=0.7, ls='--')
    ax.set_title('Heat shock response gene expression across HSP family chemotypes\n'
                  '(LINCS L1000 CD-signature percentile ranks; rows ordered by composite HSR score)',
                  fontsize=11.5, pad=10)
    plt.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        plt.savefig(FIG / f'hsr_heatmap.{ext}', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved hsr_heatmap")

# Figure 3: HSR vs 3D activity
if PER_COMPOUND.exists() and len(hs_df):
    pc = pd.read_csv(PER_COMPOUND)
    pc['_dn'] = pc['Drug_name'].fillna('').astype(str).str.upper().str.strip()
    merged_3d = hs_df.copy()
    merged_3d['_dn'] = merged_3d['compound'].str.upper()
    merged_3d = merged_3d.merge(pc[['_dn', 'activity_RMS_3D', 'rank_3D_RMS_pct']],
                                  on='_dn', how='left')
    valid = merged_3d.dropna(subset=['activity_RMS_3D'])
    if len(valid) >= 3:
        try:
            rho, pval = stats.spearmanr(valid['HSR_mean_pctrank'],
                                          valid['activity_RMS_3D'])
        except Exception:
            rho, pval = np.nan, np.nan
        fig, ax = plt.subplots(figsize=(10, 8))
        for _, r in valid.iterrows():
            color = CHEMOTYPE_PALETTE.get(r['chemotype'], '#888')
            marker = '*' if r['is_hit_3D'] else 'o'
            size = 280 if r['is_hit_3D'] else 130
            edge = 'black' if r['is_hit_3D'] else 'white'
            lw = 2 if r['is_hit_3D'] else 0.8
            ax.scatter(r['HSR_mean_pctrank'], r['activity_RMS_3D'],
                        s=size, color=color, edgecolor=edge,
                        linewidth=lw, marker=marker, zorder=3)
            ax.annotate(r['compound'],
                         (r['HSR_mean_pctrank'], r['activity_RMS_3D']),
                         xytext=(8, 5), textcoords='offset points',
                         fontsize=10,
                         fontweight='bold' if r['is_hit_3D'] else 'normal')
        ax.axvline(50, color='black', lw=0.6, ls='--', alpha=0.6,
                    label='Neutral (rank 50)')
        ax.axvline(75, color='#888', lw=0.5, ls=':', alpha=0.4)
        ax.axhline(2.2, color='#888', ls='--', lw=0.7, alpha=0.5,
                    label='P95 3D activity threshold')
        ax.set_xlabel('LINCS L1000 HSR composite percentile rank (transcriptomic)\n'
                       '50 = neutral, > 75 = enriched heat shock response',
                       fontsize=11)
        ax.set_ylabel('3D-MEDINA HepG2 activity_RMS (phenotypic)', fontsize=12)
        ax.set_title(f'Transcriptomic heat shock response vs phenotypic 3D activity\n'
                      f'Spearman rho = {rho:.2f}, p = {pval:.3g}  (n = {len(valid)})',
                      fontsize=12, fontweight='bold', loc='left', pad=10)
        legend_patches = [Patch(facecolor=col, edgecolor='black', label=ct)
                           for ct, col in CHEMOTYPE_PALETTE.items()
                           if ct in valid['chemotype'].unique()]
        ax.legend(handles=legend_patches, loc='upper left', fontsize=9)
        ax.grid(alpha=0.25)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout()
        for ext in ('png', 'pdf', 'svg'):
            plt.savefig(FIG / f'hsr_vs_3d_activity.{ext}', dpi=200,
                         bbox_inches='tight')
        plt.close()
        print(f"  Saved hsr_vs_3d_activity (Spearman rho={rho:.2f}, p={pval:.3g})")

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
summary = OUT / 'summary.txt'
with open(summary, 'w', encoding='utf-8') as f:
    f.write("LINCS L1000 transcriptomic validation - HSP family chemotypes\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Source: SigCom LINCS / LDP3 (MaayanLab; LINCS-DCIC S3 bucket)\n")
    f.write(f"Signature representation: Characteristic Direction (CD) coefficients\n")
    f.write(f"  (Clark et al. BMC Bioinformatics 2014). CD values are unit-vector\n")
    f.write(f"  loadings, naturally bounded ~[-0.06, +0.06]. The biologically\n")
    f.write(f"  meaningful quantity is the within-signature percentile rank of\n")
    f.write(f"  each gene: 50 = neutral, > 75 = enriched, < 25 = repressed.\n\n")
    f.write(f"Compounds queried:                {len(COMPOUND_QUERIES)}\n")
    f.write(f"Compounds with signature found:   {len(hs_df)}\n")
    f.write(f"Heat shock genes evaluated:       {len(HEAT_SHOCK_GENES)}\n")
    f.write(f"({', '.join(HEAT_SHOCK_GENES)})\n\n")

    f.write("HSR composite percentile rank per compound (sorted descending)\n")
    f.write("-" * 70 + "\n")
    f.write(hs_df[['compound', 'chemotype', 'cell_line', 'pert_dose',
                     'pert_time', 'HSR_mean_pctrank', 'HSR_mean_cd_raw',
                     'n_heat_shock_genes', 'is_hit_3D']].to_string(index=False))
    f.write("\n\n")

    f.write("HSR by chemotype (mean percentile rank, baseline = 50)\n")
    f.write("-" * 70 + "\n")
    for ct in CHEMOTYPE_PALETTE:
        sub = hs_df[hs_df['chemotype'] == ct]
        if not sub.empty:
            n = len(sub)
            m = sub['HSR_mean_pctrank'].mean()
            sd = sub['HSR_mean_pctrank'].std() if n > 1 else 0
            f.write(f"  {ct:55s} n={n}, mean HSR rank={m:5.1f} +/- {sd:4.1f}\n")

    # Add the ansamycin vs others stat if available
    ansamycin_hits_s = hs_df[(hs_df['chemotype'] == 'Ansamycin (HSP90)')
                              & hs_df['is_hit_3D']]
    other_hsps_s = hs_df[(hs_df['chemotype'] != 'Ansamycin (HSP90)')
                          & hs_df['chemotype'].str.contains('HSP90', na=False)]
    if len(ansamycin_hits_s) >= 2 and len(other_hsps_s) >= 2:
        try:
            u, p_mw = stats.mannwhitneyu(ansamycin_hits_s['HSR_mean_pctrank'],
                                           other_hsps_s['HSR_mean_pctrank'],
                                           alternative='greater')
            f.write("\nAnsamycin-hits vs non-ansamycin HSP90 inhibitors\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Mann-Whitney U one-sided (ansamycin > others):\n")
            f.write(f"    U = {u:.1f}, p = {p_mw:.3g}\n")
        except Exception:
            pass

print(f"\n  Saved summary: {summary}")
print(f"\n{'='*70}\nDone. Outputs in: {OUT}\n{'='*70}")
