"""
14_dili_predictor.py
======================
Evaluation of 3D HepG2 Cell Painting as an in vitro predictor of clinically
documented drug-induced liver injury (DILI).

Cross-references the EU-OPENSCREEN Bioactive Set against the FDA Liver Toxicity
Knowledge Base (LTKB) DILIrank dataset and tests whether morphological activity
in 3D HepG2 spheroids (this study) and matched 2D HepG2 monolayers
(Wolff et al. iScience 2025) predicts the FDA "vMost/vLess-DILI-Concern"
classification.

Inputs
------
data/external/DILIrank.xlsx (or .csv)
    FDA DILIrank dataset (Chen et al., Drug Discov Today 2016).
    Download from: https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-rank-dilirank-dataset
results/medina_2d_vs_3d/per_compound.csv (from script 11)
data/annotated/cp3d_library_annotated.csv (from script 10)

Outputs (saved in results/dili/)
--------------------------------
library_dili_annotated.csv    Each library compound with DILIrank class
contingency_3D_dili.csv        Fisher exact 3D-hit x DILI+
roc_metrics.csv                AUC per 2D / 3D metric, with bootstrap CI
per_hit_dili.csv               25 3D hits with DILI annotation
summary.txt                    Narrative summary
figures/
    dili_coverage_donut.png    Coverage of the library in DILIrank
    roc_2d_vs_3d.png           ROC curves comparing 2D vs 3D metrics
    dili_class_in_hits.png     DILI class distribution among hits vs non-hits
    hit_count_by_dili.png      Hit count stratified by DILI class

Methodology
-----------
DILI+ = vMost-DILI-Concern OR vLess-DILI-Concern (FDA "any clinical concern")
DILI- = vNo-DILI-Concern
Ambiguous = Ambiguous-DILI-Concern (excluded from binary tests)

1. Match library compounds to DILIrank by Drug_name (case-insensitive),
   then optionally by CAS if available.
2. Coverage report (how many of 735 ECBL bioactives are in DILIrank).
3. Fisher exact: 3D hits x DILI+ enrichment.
4. ROC analysis: each activity score (3D RMS, 2D RMS, 2D active features)
   as a continuous predictor of DILI+ status. AUC + bootstrap 95% CI.
5. DeLong test for pairwise AUC differences (optional, requires statsmodels).
6. Per-hit DILI annotation table for the 25 3D hits.

Usage
-----
    python 14_dili_predictor.py
    python 14_dili_predictor.py --dili_file path/to/custom_dili.xlsx
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

BASE = Path(r'C:\Users\Ianezi\Documents\CP3D')
ANALYSIS = BASE / 'analysis'
EXT = ANALYSIS / 'data' / 'external'
LIB = ANALYSIS / 'data' / 'annotated' / 'cp3d_library_annotated.csv'
PER_COMPOUND = ANALYSIS / 'results' / 'medina_2d_vs_3d' / 'per_compound.csv'
OUT = ANALYSIS / 'results' / 'dili'
FIG = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--dili_file', default=None,
                    help='Path to DILIrank file (xlsx or csv). Default: auto-detect in data/external/')
parser.add_argument('--n_bootstrap', type=int, default=1000,
                    help='Bootstrap iterations for AUC confidence intervals (default 1000)')
parser.add_argument('--random_state', type=int, default=42)
args = parser.parse_args()
rng = np.random.default_rng(args.random_state)

print(f"\n{'='*70}\n=== DILI prediction analysis: 3D HepG2 Cell Painting ===\n{'='*70}")

# ---------------------------------------------------------------------------
# 1. Locate and load DILIrank
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n1. Loading DILIrank (FDA LTKB)\n{'-'*70}")

if args.dili_file:
    dili_path = Path(args.dili_file)
else:
    candidates = list(EXT.glob('DILIrank*.xlsx')) + list(EXT.glob('DILIrank*.csv')) \
               + list(EXT.glob('*dili*.xlsx')) + list(EXT.glob('*DILI*.xlsx'))
    if not candidates:
        print(f"  ERROR: No DILIrank file found in {EXT}")
        print(f"  Download from: https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-rank-dilirank-dataset")
        print(f"  Save as: {EXT / 'DILIrank.xlsx'}")
        sys.exit(1)
    dili_path = candidates[0]

print(f"  File: {dili_path.name}")

def smart_load_dili(path, max_header_search=5):
    """Auto-detect header row by searching first rows for known DILIrank columns."""
    expected_patterns = ['compound', 'dili', 'concern', 'severity', 'label',
                          'generic_name', 'cas']
    for h in range(max_header_search):
        try:
            if path.suffix.lower() == '.xlsx':
                df = pd.read_excel(path, header=h)
            else:
                df = pd.read_csv(path, header=h)
            cols_lower = ' '.join(str(c).lower() for c in df.columns)
            n_hits = sum(1 for p in expected_patterns if p in cols_lower)
            if n_hits >= 2:   # at least 2 DILIrank-typical columns found
                return df, h
        except Exception:
            continue
    # Fall back to header=0 if nothing matched
    if path.suffix.lower() == '.xlsx':
        return pd.read_excel(path), 0
    return pd.read_csv(path), 0

dili, header_row_used = smart_load_dili(dili_path)
print(f"  Header row auto-detected at: {header_row_used}")
print(f"  Loaded: {len(dili)} rows x {dili.shape[1]} columns")
print(f"  Columns: {list(dili.columns)[:10]}")

# Auto-detect key columns
def find_col(df, candidates):
    lower_to_real = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        for key, real in lower_to_real.items():
            if cand.lower() in key:
                return real
    return None

name_col = find_col(dili, ['Compound Name', 'Generic_Name', 'Generic Name', 'Drug Name', 'Name'])
concern_col = find_col(dili, ['vDILIConcern', 'DILI Concern', 'DILIConcern', 'Concern'])
severity_col = find_col(dili, ['Severity Class', 'Severity_Class', 'Severity'])
cas_col = find_col(dili, ['CAS Number', 'CAS', 'CAS_Number'])

print(f"  Detected columns:")
print(f"    name:     {name_col}")
print(f"    concern:  {concern_col}")
print(f"    severity: {severity_col}")
print(f"    cas:      {cas_col}")

if not name_col or not concern_col:
    print(f"  ERROR: required columns (compound name + DILI concern) not detected.")
    print(f"  Available: {list(dili.columns)}")
    sys.exit(1)

dili = dili.rename(columns={name_col: '_dili_name', concern_col: '_dili_concern'})
if severity_col:
    dili = dili.rename(columns={severity_col: '_dili_severity'})
if cas_col:
    dili = dili.rename(columns={cas_col: '_dili_cas'})

# Normalize drug names
dili['_name_upper'] = dili['_dili_name'].fillna('').astype(str).str.upper().str.strip()
# Strip trailing salt-form descriptors like "HYDROCHLORIDE", "SODIUM" for matching
SALT_FORMS = ['HYDROCHLORIDE', 'SODIUM', 'CALCIUM', 'POTASSIUM', 'MESYLATE',
              'SUCCINATE', 'CITRATE', 'PHOSPHATE', 'SULFATE', 'TARTRATE',
              'MALEATE', 'BESYLATE', 'TOSYLATE', 'FUMARATE']
def strip_salt(name):
    for s in SALT_FORMS:
        if name.endswith(' ' + s):
            name = name[:-(len(s)+1)].strip()
    return name
dili['_name_norm'] = dili['_name_upper'].apply(strip_salt)

# DILI concern distribution
print(f"\n  DILI concern distribution in DILIrank:")
for cls, n in dili['_dili_concern'].value_counts().items():
    print(f"    {cls}: {n}")

# Binary DILI flag
def binary_dili(c):
    if pd.isna(c):
        return None
    c_low = str(c).lower()
    if 'most' in c_low or 'less' in c_low:
        return 1   # DILI+
    if 'no-dili' in c_low or c_low.endswith('no'):
        return 0   # DILI-
    return -1      # Ambiguous (excluded from binary tests)
dili['_dili_binary'] = dili['_dili_concern'].apply(binary_dili)

# ---------------------------------------------------------------------------
# 2. Cross with library
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n2. Matching library compounds to DILIrank\n{'-'*70}")

lib = pd.read_csv(LIB)
pc = pd.read_csv(PER_COMPOUND)
merged = lib.merge(pc[['EOS_id', 'activity_RMS_3D', 'activity_RMS_2D',
                         'n_features_active', 'is_hit_3D']],
                    on='EOS_id', how='inner', suffixes=('_lib', ''))
# Si la columna is_hit_3D viene duplicada, usar la de per_compound
if 'is_hit_3D_lib' in merged.columns:
    merged = merged.drop(columns=['is_hit_3D_lib'])
print(f"  Library + 2D/3D metrics: {len(merged)} compounds")

merged['_drug_norm'] = merged['Drug_name'].fillna('').astype(str).str.upper().str.strip()
merged['_drug_norm'] = merged['_drug_norm'].apply(strip_salt)

# Match by normalized name (primary)
dili_lookup_name = dili.drop_duplicates(subset=['_name_norm']).set_index('_name_norm')
cols_to_pull = ['_dili_concern', '_dili_binary']
if '_dili_severity' in dili.columns:
    cols_to_pull.append('_dili_severity')
if '_dili_cas' in dili.columns:
    cols_to_pull.append('_dili_cas')
matched = merged.merge(
    dili_lookup_name[cols_to_pull],
    left_on='_drug_norm', right_index=True, how='left'
)
n_by_name = matched['_dili_concern'].notna().sum()
print(f"  Matched by drug name:    {n_by_name}/{len(merged)} "
      f"({100*n_by_name/len(merged):.1f}%)")

# ------------------- Additional pass: match by InChIKey via PubChem-enriched file
ENRICHED = EXT / 'DILIrank_pubchem_enriched.csv'
if ENRICHED.exists():
    enriched = pd.read_csv(ENRICHED)
    enriched_ik = (enriched.dropna(subset=['InChIKey'])
                            .drop_duplicates(subset=['InChIKey']))
    raw_concern_col = None
    for c in enriched_ik.columns:
        if 'concern' in str(c).lower():
            raw_concern_col = c
            break
    if raw_concern_col is None:
        print(f"  WARNING: no DILI concern column in enriched file. Skip IK match.")
    else:
        # Exact InChIKey lookup
        ik_to_concern = dict(zip(enriched_ik['InChIKey'], enriched_ik[raw_concern_col]))
        # Skeleton InChIKey lookup (first 14 chars = connectivity, ignores stereo)
        # Use the first compound's concern when multiple stereoisomers exist.
        skel_to_concern = {}
        for ik, c in ik_to_concern.items():
            skel = str(ik).split('-')[0]
            skel_to_concern.setdefault(skel, c)

        # Library InChIKey columns to try, in priority order
        ik_cols = [c for c in ['InChIKey', 'InChIKey_rdkit', 'InChIKey_moa']
                    if c in matched.columns]

        # Pass 1: exact InChIKey match
        added_exact = 0
        missing_mask = matched['_dili_concern'].isna()
        for idx in matched[missing_mask].index:
            for ik_col in ik_cols:
                ik_val = matched.at[idx, ik_col]
                if pd.notna(ik_val) and ik_val in ik_to_concern:
                    matched.at[idx, '_dili_concern'] = ik_to_concern[ik_val]
                    matched.at[idx, '_dili_binary'] = binary_dili(ik_to_concern[ik_val])
                    added_exact += 1
                    break
        # Pass 2: skeleton InChIKey (same connectivity, different stereo)
        added_skel = 0
        missing_mask = matched['_dili_concern'].isna()
        for idx in matched[missing_mask].index:
            for ik_col in ik_cols:
                ik_val = matched.at[idx, ik_col]
                if pd.notna(ik_val):
                    skel = str(ik_val).split('-')[0]
                    if skel in skel_to_concern:
                        matched.at[idx, '_dili_concern'] = skel_to_concern[skel]
                        matched.at[idx, '_dili_binary'] = binary_dili(skel_to_concern[skel])
                        added_skel += 1
                        break
        print(f"  Matched by exact InChIKey:    +{added_exact}")
        print(f"  Matched by skeleton InChIKey: +{added_skel}  (same molecule, different stereo)")
else:
    print(f"  (No PubChem-enriched file found. Run 14a_enrich_dili_pubchem.py")
    print(f"   to add InChIKey-based matching and improve coverage.)")

n_in_dili = matched['_dili_concern'].notna().sum()
print(f"\n  TOTAL matched (name + InChIKey): {n_in_dili}/{len(merged)} "
      f"({100*n_in_dili/len(merged):.1f}%)")

# Coverage of 3D hits
n_hits_in_dili = matched.loc[matched['is_hit_3D'], '_dili_concern'].notna().sum()
n_hits_total = int(matched['is_hit_3D'].sum())
print(f"  Of {n_hits_total} 3D hits: {n_hits_in_dili} in DILIrank ({100*n_hits_in_dili/n_hits_total:.1f}%)")

# Save annotated library
matched_out = matched[['EOS_id', 'Drug_name', 'MoA', 'Target_name',
                        'is_hit_3D', 'activity_RMS_3D', 'activity_RMS_2D',
                        'n_features_active',
                        '_dili_concern', '_dili_binary']].copy()
matched_out.columns = ['EOS_id', 'Drug_name', 'MoA', 'Target_name',
                        'is_hit_3D', 'activity_RMS_3D', 'activity_RMS_2D',
                        'n_features_active_2D',
                        'DILIrank_concern', 'DILI_binary']
matched_out.to_csv(OUT / 'library_dili_annotated.csv', index=False)

# ---------------------------------------------------------------------------
# 3. Fisher exact: 3D hit x DILI+
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n3. Fisher exact: 3D hit x DILI+ enrichment\n{'-'*70}")

# Only compounds with valid binary DILI annotation
binary_set = matched[matched['_dili_binary'].isin([0, 1])].copy()
print(f"  Library with binary DILI annotation: {len(binary_set)}")

ct = pd.crosstab(binary_set['is_hit_3D'], binary_set['_dili_binary'].astype(int),
                  rownames=['Hit 3D'], colnames=['DILI+'])
# Garantizar 2x2 (reindex con fill maneja casos donde una clase no aparece)
ct = ct.reindex(index=[False, True], columns=[0, 1], fill_value=0)
print(f"\n  Contingency table:")
print(ct.to_string())
try:
    odds, pval = stats.fisher_exact(ct.values, alternative='greater')
    print(f"\n  Fisher exact (one-sided 'greater'): OR = {odds:.3f}, p = {pval:.4g}")
except Exception as e:
    print(f"  Fisher exact failed: {e}")
    odds, pval = np.nan, np.nan

ct.to_csv(OUT / 'contingency_3D_dili.csv')
pd.DataFrame({
    'metric': ['Fisher_OR', 'Fisher_p_one_sided_greater', 'n_compounds',
               'n_hits_3D', 'n_DILI_pos', 'n_overlap_hits_DILI'],
    'value': [odds, pval,
              int(len(binary_set)),
              int(binary_set['is_hit_3D'].sum()),
              int((binary_set['_dili_binary'] == 1).sum()),
              int(((binary_set['is_hit_3D']) & (binary_set['_dili_binary'] == 1)).sum())]
}).to_csv(OUT / 'fisher_test.csv', index=False)

# ---------------------------------------------------------------------------
# 4. ROC analysis: continuous score predicting DILI+
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n4. ROC analysis (DILI+ prediction)\n{'-'*70}")

def auc_with_bootstrap(y_true, scores, n_iter=1000, seed=42):
    """AUC + 95% bootstrap CI."""
    rng_local = np.random.default_rng(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    # Drop NaNs
    valid = ~np.isnan(scores)
    y_true = y_true[valid]
    scores = scores[valid]
    if len(np.unique(y_true)) < 2:
        return np.nan, (np.nan, np.nan)
    auc = roc_auc_score(y_true, scores)
    boot = []
    for _ in range(n_iter):
        idx = rng_local.integers(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot.append(roc_auc_score(y_true[idx], scores[idx]))
    if not boot:
        return auc, (np.nan, np.nan)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return auc, (lo, hi)

y = binary_set['_dili_binary'].astype(int).values
metrics = {
    'activity_RMS_3D': binary_set['activity_RMS_3D'].values,
    'activity_RMS_2D': binary_set['activity_RMS_2D'].values,
    'n_features_active_2D': binary_set['n_features_active'].values,
}
auc_rows = []
roc_curves = {}
for label, scores in metrics.items():
    auc, (lo, hi) = auc_with_bootstrap(y, scores, n_iter=args.n_bootstrap,
                                         seed=args.random_state)
    valid = ~np.isnan(scores)
    fpr, tpr, _ = roc_curve(y[valid], scores[valid])
    roc_curves[label] = (fpr, tpr, auc)
    auc_rows.append({
        'metric': label,
        'AUC': auc, 'CI_low': lo, 'CI_high': hi,
        'n': int(valid.sum()),
        'n_DILI_pos': int(((binary_set['_dili_binary'] == 1) & valid).sum()),
    })
    print(f"  AUC {label:<25}: {auc:.3f}  (95% CI: {lo:.3f} – {hi:.3f}, n = {valid.sum()})")

auc_df = pd.DataFrame(auc_rows)
auc_df.to_csv(OUT / 'roc_metrics.csv', index=False)

# ---------------------------------------------------------------------------
# 5. Per-hit DILI annotation
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n5. Per-hit DILI annotation\n{'-'*70}")

hits_dili = matched[matched['is_hit_3D']].copy()
hits_dili = hits_dili.sort_values('activity_RMS_3D', ascending=False)
hits_dili_out = hits_dili[['EOS_id', 'Drug_name', 'MoA', 'Target_name',
                             'activity_RMS_3D', 'activity_RMS_2D',
                             'n_features_active',
                             '_dili_concern']].copy()
hits_dili_out.columns = ['EOS_id', 'Drug_name', 'MoA', 'Target_name',
                          'activity_RMS_3D', 'activity_RMS_2D',
                          'n_features_active_2D', 'DILIrank_concern']
hits_dili_out.to_csv(OUT / 'per_hit_dili.csv', index=False)

print(f"\n  3D hits with DILIrank annotation:")
disp = hits_dili_out[['Drug_name', 'DILIrank_concern', 'activity_RMS_3D']].copy()
disp['Drug_name'] = disp['Drug_name'].fillna('NA').str[:28]
disp['DILIrank_concern'] = disp['DILIrank_concern'].fillna('— (not in DILIrank)')
print(disp.to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
print(f"\n{'-'*70}\n6. Figures\n{'-'*70}")

# Coverage donut
fig, ax = plt.subplots(figsize=(6, 6))
cov_counts = [
    int((matched['_dili_concern'].str.contains('Most', na=False)).sum()),
    int((matched['_dili_concern'].str.contains('Less', na=False)).sum()),
    int((matched['_dili_concern'].str.contains('Ambiguous', na=False)).sum()),
    int((matched['_dili_concern'].str.contains('No-DILI', na=False) |
         matched['_dili_concern'].str.contains(r'\bNo\b', regex=True, na=False)).sum()),
    int(matched['_dili_concern'].isna().sum())
]
cov_labels = ['vMost-DILI', 'vLess-DILI', 'Ambiguous', 'vNo-DILI', 'Not in DILIrank']
cov_colors = ['#B22222', '#E69138', '#999999', '#377EB8', '#DDDDDD']
ax.pie(cov_counts, labels=cov_labels, colors=cov_colors, autopct='%1.0f%%',
        startangle=90, wedgeprops={'width': 0.45, 'edgecolor': 'white', 'linewidth': 1.5},
        textprops={'fontsize': 10})
ax.set_title(f'DILIrank coverage of the EU-OPENSCREEN Bioactive Set\n'
              f'(n = {len(matched)} compounds; {n_in_dili} mapped to DILIrank)',
              fontsize=11, pad=12)
plt.tight_layout()
plt.savefig(FIG / 'dili_coverage_donut.png', dpi=200, bbox_inches='tight')
plt.close()

# ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
colors = {'activity_RMS_3D': '#D7263D', 'activity_RMS_2D': '#1F77B4',
          'n_features_active_2D': '#2CA02C'}
for label, (fpr, tpr, auc) in roc_curves.items():
    ci = next(r for r in auc_rows if r['metric'] == label)
    ax.plot(fpr, tpr, color=colors.get(label, 'gray'), lw=2.2,
             label=f"{label}\nAUC = {auc:.3f} (95% CI {ci['CI_low']:.2f}–{ci['CI_high']:.2f})")
ax.plot([0, 1], [0, 1], '--', color='#888', lw=0.7, alpha=0.7)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title(f'ROC: morphological activity as predictor of clinical DILI+\n'
              f'(FDA DILIrank vMost+vLess vs vNo, n = {len(binary_set)} matched compounds)',
              fontsize=11)
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(FIG / 'roc_2d_vs_3d.png', dpi=200, bbox_inches='tight')
plt.close()

# DILI class distribution in hits vs non-hits
fig, ax = plt.subplots(figsize=(8, 5))
order = ['vMost-DILI-Concern', 'vLess-DILI-Concern', 'Ambiguous-DILI-Concern',
         'vNo-DILI-Concern', 'Not in DILIrank']
hit_status = matched['is_hit_3D'].map({True: '3D hit', False: 'Non-hit'})
matched['_concern_filled'] = matched['_dili_concern'].fillna('Not in DILIrank')
tab = pd.crosstab(matched['_concern_filled'], hit_status, normalize='columns') * 100
tab = tab.reindex([c for c in order if c in tab.index])
tab.plot(kind='bar', ax=ax, color=['#1F77B4', '#D7263D'], edgecolor='black', width=0.7)
ax.set_xlabel('')
ax.set_ylabel('% of compounds in group')
ax.set_title('DILIrank concern class distribution in 3D hits vs non-hits',
              fontsize=12, pad=10)
ax.legend(title='', fontsize=10)
ax.tick_params(axis='x', rotation=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(FIG / 'dili_class_in_hits.png', dpi=200, bbox_inches='tight')
plt.close()

print(f"  Figures saved in: {FIG}")

# ---------------------------------------------------------------------------
# 7. Narrative summary
# ---------------------------------------------------------------------------
summary = OUT / 'summary.txt'
with open(summary, 'w', encoding='utf-8') as f:
    f.write("3D HepG2 Cell Painting as a predictor of clinical DILI\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"DILIrank source: FDA Liver Toxicity Knowledge Base (LTKB);\n")
    f.write(f"  Chen M et al. Drug Discov Today 2016.\n")
    f.write(f"Library: EU-OPENSCREEN Bioactive Set, MEDINA 3D and 2D screen.\n\n")

    f.write("COVERAGE\n")
    f.write("--------\n")
    f.write(f"Total library compounds: {len(merged)}\n")
    f.write(f"Mapped to DILIrank by drug name: {n_in_dili} "
            f"({100*n_in_dili/len(merged):.1f}%)\n")
    f.write(f"Of {n_hits_total} 3D hits: {n_hits_in_dili} in DILIrank "
            f"({100*n_hits_in_dili/n_hits_total:.1f}%)\n\n")

    f.write("FISHER EXACT (3D hit x DILI+)\n")
    f.write("-----------------------------\n")
    f.write(ct.to_string())
    f.write(f"\nFisher OR = {odds:.3f}, p = {pval:.4g} (one-sided greater)\n\n")

    f.write("ROC ANALYSIS (DILI+ prediction)\n")
    f.write("-------------------------------\n")
    for r in auc_rows:
        f.write(f"  {r['metric']:<25}  AUC = {r['AUC']:.3f}  "
                f"(95% CI: {r['CI_low']:.3f} – {r['CI_high']:.3f}, n = {r['n']})\n")
    f.write("\n")

    f.write("3D HITS WITH DILIRANK ANNOTATION\n")
    f.write("--------------------------------\n")
    for _, r in hits_dili_out.iterrows():
        drug = str(r['Drug_name']) if pd.notna(r['Drug_name']) else r['EOS_id']
        concern = str(r['DILIrank_concern']) if pd.notna(r['DILIrank_concern']) else '— not in DILIrank'
        f.write(f"  {r['EOS_id']:<12} {drug:<28} | RMS_3D = {r['activity_RMS_3D']:.2f}  "
                f"| {concern}\n")

print(f"\n  Summary file: {summary}")
print(f"\n{'='*70}\nAnalysis complete. Outputs in: {OUT}\n{'='*70}\n")
