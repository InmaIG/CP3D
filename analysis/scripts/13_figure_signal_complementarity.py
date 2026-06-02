"""
13_figure_signal_complementarity.py
=====================================
Generate the publication figure illustrating the complementarity of 2D and 3D
Cell Painting signals in HepG2 cells (EU-OPENSCREEN Bioactive subset).

Composition (3 panels)
----------------------
A) Compound-level partition: Venn diagram of the top-25 2D vs top-25 3D
   compound sets (matched N for direct comparison).
B) Mirror enrichment plot: enriched targets in 2D (left, blue palette) vs 3D
   (right, red palette), grouped by functional family and color-coded.
   X axis: -log10(Fisher p, one-sided greater). Dot area: Fisher odds ratio.
C) Functional family summary legend.

Inputs
------
results/medina_2d_vs_3d/target_signals/target_enrichment.csv

Outputs
-------
results/medina_2d_vs_3d/target_signals/figures/signal_complementarity_figure.{png,pdf,svg}

Usage
-----
    python 13_figure_signal_complementarity.py
    python 13_figure_signal_complementarity.py --p_cutoff 0.05
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import gridspec
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['pdf.fonttype'] = 42       # editable text in PDF/SVG
plt.rcParams['svg.fonttype'] = 'none'   # editable text in SVG

BASE = Path(r'C:\Users\Ianezi\Documents\CP3D')
ANALYSIS = BASE / 'analysis'
TS = ANALYSIS / 'results' / 'medina_2d_vs_3d' / 'target_signals'
ENRICH_CSV = TS / 'target_enrichment.csv'
TOP_2D_CSV = TS / 'top25_2D_compounds.csv'
TOP_3D_CSV = TS / 'top25_3D_compounds.csv'
FIG_DIR = TS / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--p_cutoff', type=float, default=0.05,
                    help='Raw p-value cutoff for inclusion in mirror plot (default 0.05)')
parser.add_argument('--max_per_side', type=int, default=20,
                    help='Maximum targets to display per side (default 20)')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Functional categorization (manual mapping based on enriched targets)
# ---------------------------------------------------------------------------
FAMILY_PALETTE = {
    # 2D-dominant
    'Cytoskeleton (Tubulin)':      '#D7263D',
    'DNA topoisomerase':            '#F46036',
    'Drug metabolism (CYP)':        '#FFB000',
    'Drug transporters':            '#A86434',
    # 3D-dominant
    'HSP90 chaperone system':       '#1F77B4',
    'STE20 / STK kinase family':    '#2CA02C',
    'Cell cycle (CDKs)':            '#9467BD',
    'DNA repair':                   '#17BECF',
    'Histone / Epigenetics':        '#E377C2',
    'Apoptosis':                    '#8C564B',
    # Generic
    'Other regulators':             '#7F7F7F',
    'Other':                        '#BBBBBB',
}

def categorize(target_name):
    """Assign a target name to a functional family based on keyword matching."""
    name = str(target_name).lower()
    if 'tubulin' in name:
        return 'Cytoskeleton (Tubulin)'
    if 'topoisomerase' in name:
        return 'DNA topoisomerase'
    if 'cytochrome' in name or 'cyp' in name:
        return 'Drug metabolism (CYP)'
    if any(x in name for x in ['multidrug', 'atp-binding cassette',
                                'solute carrier', 'organic anion',
                                'transporter abcg']):
        return 'Drug transporters'
    if any(x in name for x in ['hsp90', 'heat shock protein', 'endoplasmin']):
        return 'HSP90 chaperone system'
    if 'serine/threonine-protein kinase' in name or 'map4k' in name:
        return 'STE20 / STK kinase family'
    if 'cyclin-dependent kinase' in name:
        return 'Cell cycle (CDKs)'
    if any(x in name for x in ['polymerase', 'apurinic', 'apyrimidinic',
                                'endonuclease', 'helicase', 'xpd']):
        return 'DNA repair'
    if any(x in name for x in ['histone', 'methyltransferase', 'menin', 'kmt']):
        return 'Histone / Epigenetics'
    if 'caspase' in name:
        return 'Apoptosis'
    if any(x in name for x in ['nuclear receptor', 'hypoxia', 'pax-',
                                'glycine receptor', 'lethal factor',
                                'paired box']):
        return 'Other regulators'
    return 'Other'

# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------
print(f"\n{'='*70}\n=== Publication figure: 2D vs 3D signal complementarity ===\n{'='*70}\n")

if not ENRICH_CSV.exists():
    print(f"ERROR: {ENRICH_CSV} not found. Run script 12 first.")
    sys.exit(1)

en = pd.read_csv(ENRICH_CSV)
en['family'] = en['category_value'].apply(categorize)
en['minus_log10_p'] = -np.log10(en['fisher_p_one_sided'].clip(lower=1e-10))

# Filter to significant in each modality
sig_2d = en[(en['modality'] == '2D') &
            (en['fisher_p_one_sided'] < args.p_cutoff)].copy()
sig_3d = en[(en['modality'] == '3D') &
            (en['fisher_p_one_sided'] < args.p_cutoff)].copy()

# Sort by p ascending then OR descending
sig_2d = sig_2d.sort_values(['fisher_p_one_sided', 'fisher_OR'],
                             ascending=[True, False]).head(args.max_per_side)
sig_3d = sig_3d.sort_values(['fisher_p_one_sided', 'fisher_OR'],
                             ascending=[True, False]).head(args.max_per_side)

print(f"  2D enriched targets (p < {args.p_cutoff}): {len(sig_2d)}")
print(f"  3D enriched targets (p < {args.p_cutoff}): {len(sig_3d)}")

# Compound overlap (for Venn panel)
top_2d_df = pd.read_csv(TOP_2D_CSV)
top_3d_df = pd.read_csv(TOP_3D_CSV)
set_2d = set(top_2d_df['EOS_id'])
set_3d = set(top_3d_df['EOS_id'])
overlap_eos = set_2d & set_3d
overlap_names = top_2d_df[top_2d_df['EOS_id'].isin(overlap_eos)]['Drug_name'].fillna(
    top_2d_df['EOS_id']).tolist()

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 3, width_ratios=[1.1, 1.4, 1.4], height_ratios=[1, 2.1],
                       wspace=0.35, hspace=0.35)
ax_venn = fig.add_subplot(gs[0, 0])
ax_summary = fig.add_subplot(gs[0, 1:])
ax_2d = fig.add_subplot(gs[1, 0:2])
ax_3d = fig.add_subplot(gs[1, 2], sharey=None)

# -- Panel A: Venn diagram ------------------------------------------------
try:
    from matplotlib_venn import venn2
    v = venn2(subsets=(len(set_2d - set_3d), len(set_3d - set_2d), len(overlap_eos)),
              set_labels=(f'Top 25\n2D MEDINA', f'Top 25\n3D MEDINA'),
              set_colors=('#1F77B4', '#D7263D'),
              ax=ax_venn)
    if v.get_label_by_id('10'): v.get_label_by_id('10').set_fontsize(12)
    if v.get_label_by_id('01'): v.get_label_by_id('01').set_fontsize(12)
    if v.get_label_by_id('11'): v.get_label_by_id('11').set_fontsize(12)
except ImportError:
    counts = [len(set_2d - set_3d), len(overlap_eos), len(set_3d - set_2d)]
    labels = ['2D-only', 'Shared', '3D-only']
    colors_bar = ['#1F77B4', '#984EA3', '#D7263D']
    bars = ax_venn.bar(labels, counts, color=colors_bar,
                       edgecolor='black', linewidth=0.7)
    for bar, n in zip(bars, counts):
        ax_venn.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.4, str(n),
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax_venn.set_ylabel('Number of compounds', fontsize=10)
    ax_venn.set_ylim(0, max(counts) * 1.15)
    ax_venn.spines['top'].set_visible(False)
    ax_venn.spines['right'].set_visible(False)
    ax_venn.tick_params(axis='both', labelsize=10)
ax_venn.set_title('A. Compound-level partition', fontsize=13, fontweight='bold',
                  loc='left', pad=10)

# -- Panel summary box ----------------------------------------------------
ax_summary.axis('off')
ax_summary.set_title('B. Most active compounds in each modality', fontsize=13,
                     fontweight='bold', loc='left', pad=10)
ytxt = 0.95
ax_summary.text(0.02, ytxt, 'TOP 5 in 2D (by active feature count):',
                fontsize=10, fontweight='bold', color='#1F77B4',
                transform=ax_summary.transAxes)
ytxt -= 0.10
for _, r in top_2d_df.head(5).iterrows():
    drug = str(r['Drug_name']) if pd.notna(r['Drug_name']) else r['EOS_id']
    moa = str(r['MoA']) if pd.notna(r['MoA']) else 'NA'
    ax_summary.text(0.02, ytxt, f"{drug:<22}  {moa[:32]}",
                    fontsize=9, family='monospace',
                    transform=ax_summary.transAxes)
    ytxt -= 0.085

ytxt -= 0.04
ax_summary.text(0.02, ytxt, 'TOP 5 3D hits (CP3D pipeline):',
                fontsize=10, fontweight='bold', color='#D7263D',
                transform=ax_summary.transAxes)
ytxt -= 0.10
for _, r in top_3d_df.head(5).iterrows():
    drug = str(r['Drug_name']) if pd.notna(r['Drug_name']) else r['EOS_id']
    moa = str(r['MoA']) if pd.notna(r['MoA']) else 'NA'
    ax_summary.text(0.02, ytxt, f"{drug:<22}  {moa[:32]}",
                    fontsize=9, family='monospace',
                    transform=ax_summary.transAxes)
    ytxt -= 0.085

# Annotate the shared compounds at bottom
ax_summary.text(0.02, ytxt - 0.02,
                f"Shared (n={len(overlap_eos)}): {', '.join(overlap_names)}",
                fontsize=9, style='italic', color='#5B2C6F',
                transform=ax_summary.transAxes)

# -- Panel C: Mirror enrichment plot (2D left, 3D right) ------------------
def plot_side(ax, df_side, side, max_p):
    """Plot a horizontal bar/dot enrichment chart for one modality."""
    if len(df_side) == 0:
        ax.text(0.5, 0.5, 'No significant targets',
                ha='center', va='center', transform=ax.transAxes)
        return
    # Group by family, then sort within family
    df_side = df_side.sort_values(['family', 'minus_log10_p'],
                                   ascending=[True, True])
    y = np.arange(len(df_side))
    colors = df_side['family'].map(FAMILY_PALETTE).fillna('#BBBBBB')
    # Dot size proportional to log(OR + 1)
    sizes = 20 + 40 * np.log10(np.clip(df_side['fisher_OR'].fillna(1), 1, 1000))
    x = df_side['minus_log10_p'].values
    if side == '2D':
        x = -x  # plot left
    ax.scatter(x, y, s=sizes, c=colors, edgecolor='black', linewidth=0.4,
               alpha=0.92, zorder=3)
    # Horizontal connector lines
    for yi, xi, ci in zip(y, x, colors):
        ax.plot([0, xi], [yi, yi], color=ci, lw=0.7, alpha=0.5, zorder=1)
    # Y labels (target names, shortened)
    labels = df_side['category_value'].apply(
        lambda s: s if len(s) <= 45 else s[:42] + '...').values
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    if side == '2D':
        ax.set_xlim(-max_p * 1.05, 0.5)
    else:
        ax.set_xlim(-0.5, max_p * 1.05)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('-log$_{10}$(p)', fontsize=10)
    ax.tick_params(axis='x', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='x', alpha=0.25)

max_p = max(sig_2d['minus_log10_p'].max() if len(sig_2d) else 0,
            sig_3d['minus_log10_p'].max() if len(sig_3d) else 0,
            1.5)

plot_side(ax_2d, sig_2d, '2D', max_p)
plot_side(ax_3d, sig_3d, '3D', max_p)

# Move 3D y labels to the right side for mirror effect
ax_3d.yaxis.tick_right()
ax_3d.yaxis.set_label_position("right")

# Panel C header (only on the 2D axes since it spans both visually) + modality subtitle
ax_2d.set_title(
    'C. Target enrichment in top-25 compounds (Fisher exact, p < 0.05)\n'
    f'\n2D MEDINA HepG2  ·  n = {len(sig_2d)} targets enriched',
    fontsize=11, color='#1F77B4', fontweight='bold', loc='left', pad=14
)
ax_3d.set_title(
    f'\n\n3D MEDINA HepG2  ·  n = {len(sig_3d)} targets enriched',
    fontsize=11, color='#D7263D', fontweight='bold', loc='left', pad=14
)

# Functional family legend
families_used = sorted(set(sig_2d['family'].tolist() + sig_3d['family'].tolist()),
                       key=lambda x: list(FAMILY_PALETTE.keys()).index(x)
                                      if x in FAMILY_PALETTE else 99)
legend_patches = [Patch(facecolor=FAMILY_PALETTE.get(f, '#BBBBBB'),
                         edgecolor='black', linewidth=0.4, label=f)
                  for f in families_used]
fig.legend(handles=legend_patches, loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, -0.04), fontsize=9, frameon=False,
           title='Functional family', title_fontsize=10)

# Main figure title
fig.suptitle(
    '2D and 3D HepG2 Cell Painting capture complementary bioactivity in the '
    'EU-OPENSCREEN Bioactive Set',
    fontsize=13, fontweight='bold', y=1.005)
fig.text(
    0.5, 0.978,
    'Only 2 of 25 top compounds overlap between modalities. '
    '2D enriches for cytoskeletal and metabolic perturbation; '
    '3D enriches for chaperone and kinase-signaling networks.',
    ha='center', fontsize=10.5, style='italic', color='#444')

plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.97])

# Save
for ext in ('png', 'pdf', 'svg'):
    out = FIG_DIR / f'signal_complementarity_figure.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f"  Saved: {out}")
plt.close()

print(f"\n{'='*70}\nDone.\n{'='*70}")
