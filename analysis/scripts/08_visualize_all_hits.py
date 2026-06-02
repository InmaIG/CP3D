"""
08_visualize_all_hits.py
========================
Visualizacion integradora de los 35 hits 3D de las 3 placas.
Genera plots multi-panel con diferentes vistas:

1. Activity x Area, todos los hits anotados
2. Activity x Replicate_corr, todos los hits
3. Distribucion de fenotipos por placa
4. Tabla resumida de los 35 hits
5. Heatmap de hits por target compartido (multi-plate convergence)
6. UMAP/PCA de hits con perfiles morfologicos completos

Uso:
    python 08_visualize_all_hits.py
"""

import sys
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

BASE = Path(r'C:\Users\Ianezi\Documents\CP3D\analysis')
PROC_DIR = BASE / 'data' / 'processed'
HITS_DIR = BASE / 'results' / 'hits_summary'
OUT_DIR = BASE / 'results' / 'integrated_hits'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLATES = ['C2386', 'C2387', 'C2388']

# Plate colors (visual distintive)
PLATE_COLORS = {
    'C2386': '#E74C3C',  # red
    'C2387': '#3498DB',  # blue
    'C2388': '#27AE60',  # green
}

# Phenotype colors
PHENOTYPE_COLORS = {
    'expanded': '#E67E22',         # orange
    'shrunken': '#9B59B6',    # purple
    'stable': '#34495E',  # dark gray
    'unknown': '#95A5A6',              # light gray
}

print(f"\n{'='*70}\n=== VISUALIZE ALL HITS — INTEGRATED VIEW ===\n{'='*70}")

# ---------------- Load hits with chemistry from each plate ----------------
all_hits = []
for plate in PLATES:
    fp = HITS_DIR / f'{plate}_hits_with_chemistry.csv'
    if not fp.exists():
        print(f"  WARNING: {fp} not found, skipping")
        continue
    df = pd.read_csv(fp)
    df['Plate'] = plate
    all_hits.append(df)

if not all_hits:
    print("ERROR: No hits files found")
    sys.exit(1)

hits = pd.concat(all_hits, ignore_index=True)
print(f"\nTotal hits loaded: {len(hits)} (across {len(PLATES)} plates)")
print(f"  By plate: {dict(hits['Plate'].value_counts())}")
print(f"  By phenotype: {dict(hits['Phenotype'].value_counts())}")
print(f"  By category: {dict(hits['Hit_category'].value_counts())}")

# Standardize columns
for c in ['Activity_score', 'Replicate_correlation', 'AreaShape_Area_zscore', 'N_targets']:
    if c in hits.columns:
        hits[c] = pd.to_numeric(hits[c], errors='coerce')

# ============================================================
# PLOT 1: Master scatter — Activity x Area, all 35 hits annotated
# ============================================================
print(f"\n{'-'*70}\nPlot 1: Master scatter (Activity x Area)\n{'-'*70}")

fig, ax = plt.subplots(figsize=(15, 9))

# Determine Y range based on actual data
y_min = max(1.5, hits['Activity_score'].min() - 0.3)
y_max = hits['Activity_score'].max() + 0.5
x_min = hits['AreaShape_Area_zscore'].min() - 0.5
x_max = hits['AreaShape_Area_zscore'].max() + 0.5

# Plot each hit
for _, row in hits.iterrows():
    plate = row['Plate']
    phenotype = row['Phenotype']
    
    color = PHENOTYPE_COLORS.get(phenotype, '#95A5A6')
    edge = PLATE_COLORS.get(plate, 'black')
    
    size = 250 if row['Hit_category'] == 'confirmed_hit' else 150
    marker = 'o' if row['Hit_category'] == 'confirmed_hit' else '^'
    
    area_val = row['AreaShape_Area_zscore'] if pd.notna(row['AreaShape_Area_zscore']) else 0
    
    ax.scatter(area_val, row['Activity_score'],
               c=color, s=size, alpha=0.85, marker=marker,
               edgecolors=edge, linewidths=2.5, zorder=3)
    
    label = str(row.get('Drug_name', '')) if pd.notna(row.get('Drug_name', '')) else row['EOS_id']
    if label == 'nan' or label == '':
        label = row['EOS_id']
    if len(label) > 16:
        label = label[:13] + '...'
    
    # Smart vertical offset based on density
    offset_x = 9
    offset_y = 0
    ax.annotate(label, (area_val, row['Activity_score']),
                xytext=(offset_x, offset_y), textcoords='offset points',
                fontsize=8, alpha=0.9, ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='none'))

# Reference lines
ax.axvline(0, color='gray', linestyle=':', alpha=0.5, zorder=1)
ax.axvline(1, color='#E67E22', linestyle='--', alpha=0.4, zorder=1)
ax.axvline(-1, color='#9B59B6', linestyle='--', alpha=0.4, zorder=1)

# Background zones with proper alpha
ax.axvspan(1, x_max, alpha=0.06, color='#E67E22', zorder=0)
ax.axvspan(x_min, -1, alpha=0.06, color='#9B59B6', zorder=0)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Zone labels at top of plot
ax.text((1 + x_max) / 2, y_min + 0.15, 'EXPANDED\n(swelling)',
        fontsize=11, fontweight='bold', color='#E67E22', alpha=0.7,
        ha='center', va='bottom')
ax.text((x_min - 1) / 2 - 0.5, y_min + 0.15, 'SHRUNKEN\n(shrinking)',
        fontsize=11, fontweight='bold', color='#9B59B6', alpha=0.7,
        ha='center', va='bottom')

ax.set_xlabel('AreaShape_Area (z-score consensus)\n← shrunken (Area < 0)  ·········  expanded (Area > 0) →',
              fontsize=11)
ax.set_ylabel('Activity score\n(distance from DMSO consensus)', fontsize=11)
ax.set_title(f'All confirmed hits ({len(hits)}) across 3 plates\nphenotype × plate origin',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(alpha=0.3)

# Legends: place all in upper right and lower right (no overlap with data)
phenotype_legend = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=PHENOTYPE_COLORS['expanded'],
           markersize=12, label='Expanded', markeredgecolor='black', markeredgewidth=1),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=PHENOTYPE_COLORS['shrunken'],
           markersize=12, label='Shrunken', markeredgecolor='black', markeredgewidth=1),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=PHENOTYPE_COLORS['stable'],
           markersize=12, label='Stable', markeredgecolor='black', markeredgewidth=1),
]
plate_legend = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='white',
           markersize=12, label=f'C2386 ({(hits["Plate"]=="C2386").sum()} hits)',
           markeredgecolor=PLATE_COLORS['C2386'], markeredgewidth=2.5),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='white',
           markersize=12, label=f'C2387 ({(hits["Plate"]=="C2387").sum()} hits)',
           markeredgecolor=PLATE_COLORS['C2387'], markeredgewidth=2.5),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='white',
           markersize=12, label=f'C2388 ({(hits["Plate"]=="C2388").sum()} hits)',
           markeredgecolor=PLATE_COLORS['C2388'], markeredgewidth=2.5),
]
shape_legend = [
    Line2D([0],[0], marker='o', color='gray', markerfacecolor='gray',
           markersize=12, label='Confirmed hit', linestyle='None'),
    Line2D([0],[0], marker='^', color='gray', markerfacecolor='gray',
           markersize=10, label='Highly toxic (1 rep)', linestyle='None'),
]

# Place legends to right of plot
l1 = ax.legend(handles=phenotype_legend, loc='upper left',
               bbox_to_anchor=(1.01, 1.0), title='Phenotype', fontsize=9)
ax.add_artist(l1)
l2 = ax.legend(handles=plate_legend, loc='upper left',
               bbox_to_anchor=(1.01, 0.7), title='Plate (border color)', fontsize=9)
ax.add_artist(l2)
ax.legend(handles=shape_legend, loc='upper left',
          bbox_to_anchor=(1.01, 0.4), title='Hit type', fontsize=9)

plt.tight_layout()
plot1 = OUT_DIR / 'all_hits_master_scatter.png'
plt.savefig(plot1, dpi=150, bbox_inches='tight', facecolor='white')
print(f"  Saved: {plot1}")
plt.close()

# ============================================================
# PLOT 2: Activity x Reproducibility, all hits
# ============================================================
print(f"\n{'-'*70}\nPlot 2: Activity x Reproducibility\n{'-'*70}")

fig, ax = plt.subplots(figsize=(13, 9))

for _, row in hits.iterrows():
    plate = row['Plate']
    phenotype = row['Phenotype']
    color = PHENOTYPE_COLORS.get(phenotype, '#95A5A6')
    edge = PLATE_COLORS.get(plate, 'black')
    size = 250 if row['Hit_category'] == 'confirmed_hit' else 150
    
    rep_corr = row['Replicate_correlation']
    if pd.isna(rep_corr):
        rep_corr = -0.05  # Place "highly toxic" hits at the bottom
    
    ax.scatter(row['Activity_score'], rep_corr,
               c=color, s=size, alpha=0.85,
               edgecolors=edge, linewidths=2.5, zorder=3)
    
    label = str(row.get('Drug_name', '')) if pd.notna(row.get('Drug_name', '')) else row['EOS_id']
    if label == 'nan' or label == '':
        label = row['EOS_id']
    if len(label) > 16:
        label = label[:13] + '...'
    
    ax.annotate(label, (row['Activity_score'], rep_corr),
                xytext=(8, 0), textcoords='offset points',
                fontsize=8, alpha=0.85, va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

ax.axhline(0.3, color='green', linestyle='--', alpha=0.5, label='Reproducibility threshold (0.3)')
ax.axhline(0, color='black', linestyle=':', alpha=0.3)

ax.set_xlabel('Activity score', fontsize=11)
ax.set_ylabel('Replicate correlation\n(NaN = highly toxic, only 1 replicate)', fontsize=11)
ax.set_title(f'Hits by activity vs reproducibility (n={len(hits)})',
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plot2 = OUT_DIR / 'all_hits_activity_vs_reproducibility.png'
plt.savefig(plot2, dpi=150, bbox_inches='tight', facecolor='white')
print(f"  Saved: {plot2}")
plt.close()

# ============================================================
# PLOT 3: Phenotype distribution by plate
# ============================================================
print(f"\n{'-'*70}\nPlot 3: Phenotype distribution by plate\n{'-'*70}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Stacked bar chart
ax = axes[0]
phenotype_by_plate = pd.crosstab(hits['Plate'], hits['Phenotype'])
# Reorder phenotypes for visual consistency
phenotype_order = ['expanded', 'shrunken', 'stable', 'unknown']
phenotype_by_plate = phenotype_by_plate.reindex(columns=[p for p in phenotype_order if p in phenotype_by_plate.columns])

bottom = np.zeros(len(phenotype_by_plate))
for phenotype in phenotype_by_plate.columns:
    color = PHENOTYPE_COLORS.get(phenotype, '#95A5A6')
    counts = phenotype_by_plate[phenotype].values
    ax.bar(phenotype_by_plate.index, counts, bottom=bottom,
           color=color, label=phenotype, alpha=0.85, edgecolor='black', linewidth=1)
    # Add count labels
    for i, count in enumerate(counts):
        if count > 0:
            ax.text(i, bottom[i] + count/2, str(int(count)),
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    bottom += counts

ax.set_ylabel('Number of hits', fontsize=11)
ax.set_xlabel('Plate', fontsize=11)
ax.set_title('Phenotype distribution per plate', fontsize=12, fontweight='bold')
ax.legend(title='Phenotype', loc='upper right', fontsize=9)
ax.grid(alpha=0.3, axis='y')

# Pie chart of all phenotypes combined
ax = axes[1]
phenotype_counts = hits['Phenotype'].value_counts()
phenotype_colors_list = [PHENOTYPE_COLORS.get(p, '#95A5A6') for p in phenotype_counts.index]
wedges, texts, autotexts = ax.pie(phenotype_counts, labels=phenotype_counts.index,
                                    colors=phenotype_colors_list, autopct='%1.0f%%',
                                    startangle=90, textprops={'fontsize': 10})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax.set_title(f'Overall phenotype distribution (n={len(hits)})',
             fontsize=12, fontweight='bold')

plt.suptitle('Phenotypic landscape of confirmed hits', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plot3 = OUT_DIR / 'all_hits_phenotype_distribution.png'
plt.savefig(plot3, dpi=150, bbox_inches='tight', facecolor='white')
print(f"  Saved: {plot3}")
plt.close()

# ============================================================
# PLOT 4: Compact summary table image (figure with text grid)
# ============================================================
print(f"\n{'-'*70}\nPlot 4: Summary table\n{'-'*70}")

# Sort hits by plate then by activity
display_hits = hits.copy()
display_hits = display_hits.sort_values(['Plate', 'Activity_score'], ascending=[True, False]).reset_index(drop=True)

fig_height = max(8, 0.32 * len(display_hits) + 2)
fig, ax = plt.subplots(figsize=(18, fig_height))
ax.axis('off')

# Build table data
table_data = []
for _, row in display_hits.iterrows():
    drug_name = str(row.get('Drug_name', '')) if pd.notna(row.get('Drug_name', '')) else ''
    if drug_name == 'nan' or drug_name == '':
        drug_name = '—'
    elif len(drug_name) > 22:
        drug_name = drug_name[:19] + '...'
    
    target = str(row.get('Target_name', '')) if pd.notna(row.get('Target_name', '')) else ''
    if target == 'nan' or target == '':
        target = '—'
    elif len(target) > 30:
        # Take first target only
        target = target.split(';')[0]
        if len(target) > 30:
            target = target[:27] + '...'
    
    moa = str(row.get('MoA', '')) if pd.notna(row.get('MoA', '')) else ''
    if moa == 'nan' or moa == '':
        moa = '—'
    elif len(moa) > 22:
        moa = moa.split(';')[0]
        if len(moa) > 22:
            moa = moa[:19] + '...'
    
    area = row.get('AreaShape_Area_zscore', np.nan)
    area_str = f"{area:+.2f}" if pd.notna(area) else 'NaN'
    
    rep_corr = row.get('Replicate_correlation', np.nan)
    rep_str = f"{rep_corr:.2f}" if pd.notna(rep_corr) else 'N/A'
    
    n_targ = row.get('N_targets', np.nan)
    n_targ_str = f"{int(n_targ)}" if pd.notna(n_targ) else '—'
    
    table_data.append([
        row['Plate'],
        row['EOS_id'],
        drug_name,
        row['Phenotype'][:18],
        f"{row['Activity_score']:.2f}",
        rep_str,
        f"{int(row['N_replicates'])}",
        area_str,
        target,
        moa,
        n_targ_str,
    ])

columns = ['Plate', 'EOS_id', 'Drug name', 'Phenotype', 'Activity', 'Rep corr',
           '# Reps', 'Area z', 'Target (first)', 'MoA', 'N targets']

# Color rows by plate
row_colors = []
for r in table_data:
    plate = r[0]
    bg_color = PLATE_COLORS[plate]
    # Light tint
    if plate == 'C2386':
        row_colors.append(['#FADBD8'] * len(columns))
    elif plate == 'C2387':
        row_colors.append(['#D6EAF8'] * len(columns))
    else:
        row_colors.append(['#D5F5E3'] * len(columns))

table = ax.table(cellText=table_data, colLabels=columns,
                 cellColours=row_colors,
                 colColours=['#34495E']*len(columns),
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.5)

# Style header
for i in range(len(columns)):
    cell = table[(0, i)]
    cell.set_text_props(color='white', fontweight='bold', fontsize=9)

# Phenotype color in cell
phen_col = columns.index('Phenotype')
for i, row in enumerate(table_data):
    phenotype = row[phen_col]
    if 'expanded' in phenotype:
        table[(i+1, phen_col)].set_facecolor('#FAD7A0')
    elif 'shrunken' in phenotype:
        table[(i+1, phen_col)].set_facecolor('#D7BDE2')

ax.set_title(f'Confirmed hits ({len(display_hits)}) — full chemistry & MoA annotation',
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plot4 = OUT_DIR / 'all_hits_summary_table.png'
plt.savefig(plot4, dpi=150, bbox_inches='tight', facecolor='white')
print(f"  Saved: {plot4}")
plt.close()

# ============================================================
# PLOT 5: Targets shared across plates (convergence)
# ============================================================
print(f"\n{'-'*70}\nPlot 5: Targets/genes shared across plates\n{'-'*70}")

# Extract genes per plate
def get_genes(s):
    if pd.isna(s) or not str(s).strip():
        return set()
    return {x.strip() for x in str(s).split(';') if x.strip()}

genes_by_plate = {}
for plate in PLATES:
    sub = hits[hits['Plate']==plate]
    all_genes = set()
    for s in sub['Gene_name'].dropna():
        all_genes.update(get_genes(s))
    genes_by_plate[plate] = all_genes
    print(f"  {plate}: {len(all_genes)} unique genes targeted")

# Find genes targeted in multiple plates
shared_2 = (genes_by_plate['C2386'] & genes_by_plate['C2387']) | \
           (genes_by_plate['C2386'] & genes_by_plate['C2388']) | \
           (genes_by_plate['C2387'] & genes_by_plate['C2388'])
shared_3 = genes_by_plate['C2386'] & genes_by_plate['C2387'] & genes_by_plate['C2388']
shared_2_only = shared_2 - shared_3

print(f"\n  Genes shared in 2+ plates: {len(shared_2)}")
print(f"  Genes shared in ALL 3 plates: {len(shared_3)}")
if shared_3:
    print(f"    {sorted(shared_3)[:20]}")

# Venn-like plot
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Bar chart of top genes shared
ax = axes[0]
gene_counts = {}
for plate, genes in genes_by_plate.items():
    for g in genes:
        gene_counts[g] = gene_counts.get(g, 0) + 1

# Genes appearing in 2+ plates
multi_plate_genes = {g: c for g, c in gene_counts.items() if c >= 2}
top_multi = sorted(multi_plate_genes.items(), key=lambda x: -x[1])[:30]

if top_multi:
    gene_names = [g for g, _ in top_multi]
    gene_n_plates = [n for _, n in top_multi]
    
    # Color by which plates it's in
    colors = []
    for g in gene_names:
        plates_with = [p for p in PLATES if g in genes_by_plate[p]]
        if len(plates_with) == 3:
            colors.append('#27AE60')  # green for all 3
        else:
            # Mix of plates
            colors.append('#F39C12')
    
    ax.barh(range(len(gene_names)), gene_n_plates, color=colors, alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(gene_names)))
    ax.set_yticklabels(gene_names, fontsize=8)
    ax.set_xlabel('Number of plates targeting this gene', fontsize=11)
    ax.set_title(f'Genes targeted by hits in ≥2 plates ({len(multi_plate_genes)} total)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks([1, 2, 3])
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='x')
    
    legend = [
        Line2D([0],[0], marker='s', color='w', markerfacecolor='#27AE60',
               markersize=12, label='In all 3 plates'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor='#F39C12',
               markersize=12, label='In 2 plates'),
    ]
    ax.legend(handles=legend, loc='lower right', fontsize=10)
else:
    ax.text(0.5, 0.5, 'No genes shared across plates', ha='center', va='center', fontsize=12)
    ax.axis('off')

# Plate composition
ax = axes[1]
counts_by_plate_phenotype = pd.crosstab(hits['Plate'], hits['Phenotype'])

# Show targets per plate (count of unique)
plates_y = list(PLATES)
genes_per_plate = [len(genes_by_plate[p]) for p in plates_y]
hits_per_plate = [(hits['Plate']==p).sum() for p in plates_y]

x = np.arange(len(plates_y))
width = 0.35

bars1 = ax.bar(x - width/2, hits_per_plate, width, label='# hits',
                color=[PLATE_COLORS[p] for p in plates_y], alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, genes_per_plate, width, label='# unique genes',
                color=[PLATE_COLORS[p] for p in plates_y], alpha=0.4, hatch='//', edgecolor='black')

for bars in [bars1, bars2]:
    for b in bars:
        height = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, height + 0.5, f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(plates_y)
ax.set_ylabel('Count')
ax.set_title('Hits and target diversity per plate', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

plt.suptitle('Target convergence across plates', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plot5 = OUT_DIR / 'all_hits_target_convergence.png'
plt.savefig(plot5, dpi=150, bbox_inches='tight', facecolor='white')
print(f"  Saved: {plot5}")
plt.close()

# Also save shared genes/targets to CSV
shared_data = []
for g, n in sorted(gene_counts.items(), key=lambda x: -x[1]):
    if n >= 2:
        plates_with = [p for p in PLATES if g in genes_by_plate[p]]
        shared_data.append({
            'gene': g,
            'n_plates': n,
            'plates': ', '.join(plates_with),
        })
if shared_data:
    pd.DataFrame(shared_data).to_csv(OUT_DIR / 'shared_targets_across_plates.csv', index=False)
    print(f"  Saved: shared_targets_across_plates.csv")

# ============================================================
# PLOT 6: Promiscuity bubble plot
# ============================================================
print(f"\n{'-'*70}\nPlot 6: Promiscuity vs Activity (bubble)\n{'-'*70}")

fig, ax = plt.subplots(figsize=(13, 9))

for _, row in hits.iterrows():
    plate = row['Plate']
    phenotype = row['Phenotype']
    
    n_targ = row.get('N_targets', np.nan)
    if pd.isna(n_targ):
        n_targ = 1
    bubble_size = max(50, min(2000, n_targ * 15))
    
    color = PHENOTYPE_COLORS.get(phenotype, '#95A5A6')
    edge = PLATE_COLORS.get(plate, 'black')
    
    rep_corr = row['Replicate_correlation']
    if pd.isna(rep_corr):
        rep_corr = 0
    
    ax.scatter(rep_corr, row['Activity_score'],
               s=bubble_size, c=color, alpha=0.6,
               edgecolors=edge, linewidths=2.5, zorder=3)
    
    label = str(row.get('Drug_name', '')) if pd.notna(row.get('Drug_name', '')) else row['EOS_id']
    if label == 'nan' or label == '':
        label = row['EOS_id']
    if len(label) > 16:
        label = label[:13] + '...'
    
    ax.annotate(label, (rep_corr, row['Activity_score']),
                xytext=(0, 0), textcoords='offset points',
                fontsize=7.5, alpha=0.9, ha='center', va='center')

ax.axvline(0.3, color='green', linestyle='--', alpha=0.5)
ax.set_xlabel('Replicate correlation (reproducibility)', fontsize=11)
ax.set_ylabel('Activity score', fontsize=11)
ax.set_title(f'Hit landscape: bubble size = N_targets (promiscuity)\nlarger = more promiscuous',
             fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Size legend
legend_sizes = [1, 10, 50, 100]
size_handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',
                        markersize=np.sqrt(max(50, min(2000, s*15))/np.pi),
                        label=f'{s} target{"s" if s>1 else ""}',
                        markeredgecolor='black', markeredgewidth=1)
                for s in legend_sizes]
ax.legend(handles=size_handles, loc='upper left', title='Bubble size = N_targets',
          fontsize=9, labelspacing=1.5)

plt.tight_layout()
plot6 = OUT_DIR / 'all_hits_promiscuity_bubble.png'
plt.savefig(plot6, dpi=150, bbox_inches='tight', facecolor='white')
print(f"  Saved: {plot6}")
plt.close()

# ============================================================
# Final summary
# ============================================================
print(f"\n{'='*70}\n=== DONE ===\n{'='*70}")
print(f"\nAll outputs in: {OUT_DIR}")
print(f"\nGenerated 6 plots + 1 CSV:")
print(f"  1. all_hits_master_scatter.png        — Activity x Area, all hits annotated")
print(f"  2. all_hits_activity_vs_reproducibility.png")
print(f"  3. all_hits_phenotype_distribution.png")
print(f"  4. all_hits_summary_table.png         — Full annotated table")
print(f"  5. all_hits_target_convergence.png    — Targets shared between plates")
print(f"  6. all_hits_promiscuity_bubble.png    — Bubble = N_targets")
print(f"  7. shared_targets_across_plates.csv")